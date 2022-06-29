from __future__ import annotations

import copy
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from qlib.backtest.account import Account
from qlib.backtest.position import BasePosition
from qlib.log import get_module_logger
from . import BaseTradeDecision, Order
from qlib.event.event import CascadeEvent, DecisionEvent, Event, EventBuffer, ExecutorPayload
from .exchange import Exchange
from .executor import _retrieve_orders_from_decision
from .utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager, get_start_end_idx
from ..strategy.base import BaseStrategy
from ..utils import init_instance_by_config


class DecisionEventType(Enum):
    YIELD_DECISION = 1


class BaseExecutorV2:
    """Base executor for trading"""

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Exchange = None,
        common_infra: CommonInfrastructure = None,
        settle_type: str = BasePosition.ST_NO,
    ) -> None:
        self._last_res: list = []
        self._last_kwargs: dict = {}

        self.time_per_step = time_per_step
        self.indicator_config = indicator_config
        self.generate_portfolio_metrics = generate_portfolio_metrics
        self.verbose = verbose
        self.track_data = track_data
        self._trade_exchange = trade_exchange
        self.level_infra = LevelInfrastructure()
        self.level_infra.reset_infra(common_infra=common_infra)
        self._settle_type = settle_type
        self.reset(start_time=start_time, end_time=end_time, common_infra=common_infra)
        if common_infra is None:
            get_module_logger("BaseExecutorV2").warning(f"`common_infra` is not set for {self}")

        # record deal order amount in one day
        self.dealt_order_amount: Dict[str, float] = defaultdict(float)
        self.deal_day = None

    @property
    def atomic(self) -> bool:
        return not issubclass(self.__class__, NestedExecutorV2)  # issubclass(A, A) is True

    def get_res_and_kwargs(self) -> Tuple[list, dict]:
        return self._last_res, self._last_kwargs

    def generate_events(self, payload: ExecutorPayload) -> List[Event]:
        return [
            CascadeEvent(payload=payload, fn=self._handler_yield_decision),
            CascadeEvent(payload=payload, fn=self._handler_pre_collect),
            CascadeEvent(payload=payload, fn=self._handler_collect),
            CascadeEvent(payload=payload, fn=self._handler_post_collect),
        ]

    def reset_common_infra(self, common_infra: CommonInfrastructure, copy_trade_account: bool = False) -> None:
        """
        reset infrastructure for trading
            - reset trade_account
        """
        if not hasattr(self, "common_infra"):
            self.common_infra = common_infra
        else:
            self.common_infra.update(common_infra)

        if common_infra.has("trade_account"):
            # NOTE: there is a trick in the code.
            # shallow copy is used instead of deepcopy.
            # 1. So positions are shared
            # 2. Others are not shared, so each level has it own metrics (portfolio and trading metrics)
            self.trade_account: Account = (
                copy.copy(common_infra.get("trade_account"))
                if copy_trade_account
                else common_infra.get("trade_account")
            )
            self.trade_account.reset(freq=self.time_per_step, port_metr_enabled=self.generate_portfolio_metrics)

    @property
    def trade_exchange(self) -> Exchange:
        """get trade exchange in a prioritized order"""
        return getattr(self, "_trade_exchange", None) or self.common_infra.get("trade_exchange")

    @property
    def trade_calendar(self) -> TradeCalendarManager:
        """
        Though trade calendar can be accessed from multiple sources, but managing in a centralized way will make the
        code easier
        """
        return self.level_infra.get("trade_calendar")

    def reset(self, common_infra: CommonInfrastructure = None, **kwargs: Any) -> None:
        """
        - reset `start_time` and `end_time`, used in trade calendar
        - reset `common_infra`, used to reset `trade_account`, `trade_exchange`, .etc
        """

        if "start_time" in kwargs or "end_time" in kwargs:
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            self.level_infra.reset_cal(freq=self.time_per_step, start_time=start_time, end_time=end_time)
        if common_infra is not None:
            self.reset_common_infra(common_infra)

    def get_level_infra(self) -> LevelInfrastructure:
        return self.level_infra

    def finished(self) -> bool:
        return self.trade_calendar.finished()

    # def execute(self, trade_decision: BaseTradeDecision, level: int = 0) -> List[object]:
    #     return_value: dict = {}
    #     for _decision in self.collect_data(trade_decision, return_value=return_value, level=level):
    #         pass
    #     return cast(list, return_value.get("execute_result"))

    def _handler_yield_decision(self, event: CascadeEvent):
        if self.track_data:
            decision_event = DecisionEvent(
                payload=event.payload.trade_decision,
                event_type=DecisionEventType.YIELD_DECISION,
            )
            event.insert_event(decision_event)

    @abstractmethod
    def _handler_collect(self, event: CascadeEvent) -> None:
        raise NotImplementedError

    def _handler_pre_collect(self, event: CascadeEvent) -> None:
        payload = event.payload
        if self.atomic and payload.trade_decision.get_range_limit(default_value=None) is not None:
            raise ValueError("atomic executor doesn't support specify `range_limit`")

        if self._settle_type != BasePosition.ST_NO:
            self.trade_account.current_position.settle_start(self._settle_type)

    def _handler_post_collect(self, event: CascadeEvent) -> None:
        payload = event.payload

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
        # Account will not be changed in this function
        self.trade_account.update_bar_end(
            trade_start_time,
            trade_end_time,
            self.trade_exchange,
            atomic=self.atomic,
            outer_trade_decision=payload.trade_decision,
            indicator_config=self.indicator_config,
            **self._last_kwargs,
        )

        self.trade_calendar.step()

        if self._settle_type != BasePosition.ST_NO:
            self.trade_account.current_position.settle_commit()

        if payload.return_value is not None:
            payload.return_value.update({"execute_result": self._last_res})

    @abstractmethod
    def get_all_executors(self) -> List[BaseExecutorV2]:
        """get all executors"""
        raise NotImplementedError


class NestedExecutorV2(BaseExecutorV2):
    def __init__(
        self,
        time_per_step: str,
        inner_executor: Union[BaseExecutorV2, dict],
        inner_strategy: Union[BaseStrategy, dict],
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        skip_empty_decision: bool = True,
        align_range_limit: bool = True,
        common_infra: CommonInfrastructure = None,
    ) -> None:
        self.inner_executor: BaseExecutorV2 = init_instance_by_config(
            inner_executor,
            common_infra=common_infra,
            accept_types=BaseExecutorV2,
        )
        self.inner_strategy: BaseStrategy = init_instance_by_config(
            inner_strategy,
            common_infra=common_infra,
            accept_types=BaseStrategy,
        )

        self._skip_empty_decision = skip_empty_decision
        self._align_range_limit = align_range_limit

        super(NestedExecutorV2, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            common_infra=common_infra,
        )

    def reset_common_infra(self, common_infra: CommonInfrastructure, copy_trade_account: bool = False) -> None:
        """
        reset infrastructure for trading
            - reset inner_strategy and inner_executor common infra
        """
        # NOTE: please refer to the docs of BaseExecutor.reset_common_infra for the meaning of `copy_trade_account`

        # The first level follow the `copy_trade_account` from the upper level
        super(NestedExecutorV2, self).reset_common_infra(common_infra, copy_trade_account=copy_trade_account)

        # The lower level have to copy the trade_account
        self.inner_executor.reset_common_infra(common_infra, copy_trade_account=True)
        self.inner_strategy.reset_common_infra(common_infra)

    def _init_sub_trading(self, trade_decision: BaseTradeDecision) -> None:
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
        self.inner_executor.reset(start_time=trade_start_time, end_time=trade_end_time)
        sub_level_infra = self.inner_executor.get_level_infra()
        self.level_infra.set_sub_level_infra(sub_level_infra)
        self.inner_strategy.reset(level_infra=sub_level_infra, outer_trade_decision=trade_decision)

    def _update_trade_decision(self, trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        # outer strategy have chance to update decision each iterator
        updated_trade_decision = trade_decision.update(self.inner_executor.trade_calendar)
        if updated_trade_decision is not None:
            trade_decision = updated_trade_decision
            # NEW UPDATE
            # create a hook for inner strategy to update outer decision
            self.inner_strategy.alter_outer_trade_decision(trade_decision)
        return trade_decision

    def _handler_execute_loop_start(self, event: CascadeEvent) -> None:
        trade_decision = self._update_trade_decision(event.payload.trade_decision)

        if self.inner_executor.finished() or (trade_decision.empty() and self._skip_empty_decision):
            event.insert_event(CascadeEvent(event.payload, self._handler_execute_loop_end))  # Finish executing
        else:
            sub_cal: TradeCalendarManager = self.inner_executor.trade_calendar
            # NOTE: make sure get_start_end_idx is after `self._update_trade_decision`
            start_idx, end_idx = get_start_end_idx(sub_cal, trade_decision)

            if not self._align_range_limit or start_idx <= sub_cal.get_trade_step() <= end_idx:
                event.insert_events([
                    # TODO: strategy gen event
                    CascadeEvent(payload=event.payload, fn=self._handler_inner_execute)
                ])
            else:
                sub_cal.step()
                event.insert_event(CascadeEvent(event.payload, self._handler_execute_loop_start))  # Continue executing

    def _handler_execute_loop_end(self, event: CascadeEvent) -> None:
        self._last_res = self._execute_result
        self._last_kwargs = {
            "inner_order_indicators": self._inner_order_indicators,
            "decision_list": self._decision_list,
        }

    def _handler_inner_execute(self, event: CascadeEvent) -> None:
        _inner_trade_decision: BaseTradeDecision = self.inner_strategy.get_decision()  # TODO: To be implemented

        event.payload.trade_decision.mod_inner_decision(_inner_trade_decision)  # propagate part of decision information

        # NOTE sub_cal.get_step_time() must be called before collect_data in case of step shifting
        sub_cal: TradeCalendarManager = self.inner_executor.trade_calendar
        self._decision_list.append((_inner_trade_decision, *sub_cal.get_step_time()))

        new_payload = ExecutorPayload(
            trade_decision=_inner_trade_decision,
            return_value=event.payload.return_value,
            level=event.payload.level + 1,
        )

        followup_events = self.inner_executor.generate_events(new_payload)
        followup_events.append(CascadeEvent(event.payload, self._handler_post_inner_execute))  # Use old payload here
        event.insert_events(followup_events)

    def _handler_post_inner_execute(self, event: CascadeEvent) -> None:
        _inner_execute_result, _ = self.inner_executor.get_res_and_kwargs()

        self.post_inner_exe_step(_inner_execute_result)
        self._execute_result.extend(_inner_execute_result)

        self._inner_order_indicators.append(
            self.inner_executor.trade_account.get_trade_indicator().get_order_indicator(raw=True),
        )

        event.insert_event(CascadeEvent(event.payload, self._handler_execute_loop_start))  # Continue executing

    def _handler_collect(self, event: CascadeEvent) -> None:
        self._execute_result = []
        self._inner_order_indicators = []
        self._decision_list = []
        self._inner_execute_result = None

        # NOTE:
        # - this is necessary to calculating the steps in sublevel
        # - more detailed information will be set into trade decision
        self._init_sub_trading(event.payload.trade_decision)

        # Pseudocode to help understanding
        #
        #   while not (inner executor finished or skip empty decision):
        #       if (in the valid range):
        #           self._inner_strategy_generate_decision
        #           self._handler_inner_execute
        #               => run inner executor loop
        #               => self._handler_post_inner_execute
        #       else:
        #           sub_cal.step()
        #
        #   self._handler_execute_loop_end
        event.insert_event(CascadeEvent(event.payload, self._handler_execute_loop_start))  # Start the loop

    def post_inner_exe_step(self, inner_exe_res: List[object]) -> None:
        pass

    def get_all_executors(self) -> List[BaseExecutorV2]:
        """get all executors, including self and inner_executor.get_all_executors()"""
        return [self, *self.inner_executor.get_all_executors()]


class FinalExecutorV2(BaseExecutorV2):
    # available trade_types
    TT_SERIAL = "serial"
    # The orders will be executed serially in a sequence
    # In each trading step, it is possible that users sell instruments first and use the money to buy new instruments
    TT_PARAL = "parallel"
    # The orders will be executed in parallel
    # In each trading step, if users try to sell instruments first and buy new instruments with money, failure will
    # occur

    def __init__(
        self,
        time_per_step: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Exchange = None,
        common_infra: CommonInfrastructure = None,
        settle_type: str = BasePosition.ST_NO,
        trade_type: str = TT_SERIAL,
    ) -> None:
        super(FinalExecutorV2, self).__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            trade_exchange=trade_exchange,
            common_infra=common_infra,
            settle_type=settle_type,
        )

        self.trade_type = trade_type

    def _get_order_iterator(self, trade_decision: BaseTradeDecision) -> List[Order]:
        """

        Parameters
        ----------
        trade_decision : BaseTradeDecision
            the trade decision given by the strategy

        Returns
        -------
        List[Order]:
            get a list orders according to `self.trade_type`
        """
        orders = _retrieve_orders_from_decision(trade_decision)

        if self.trade_type == self.TT_SERIAL:
            # Orders will be traded in a parallel way
            order_it = orders
        elif self.trade_type == self.TT_PARAL:
            # NOTE: !!!!!!!
            # Assumption: there will not be orders in different trading direction in a single step of a strategy !!!!
            # The parallel trading failure will be caused only by the conflicts of money
            # Therefore, make the buying go first will make sure the conflicts happen.
            # It equals to parallel trading after sorting the order by direction
            order_it = sorted(orders, key=lambda order: -order.direction)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return order_it

    def _update_dealt_order_amount(self, order: Order) -> None:
        """update date and dealt order amount in the day."""

        now_deal_day = self.trade_calendar.get_step_time()[0].floor(freq="D")
        if self.deal_day is None or now_deal_day > self.deal_day:
            self.dealt_order_amount = defaultdict(float)
            self.deal_day = now_deal_day
        self.dealt_order_amount[order.stock_id] += order.deal_amount

    def _handler_collect(self, event: CascadeEvent) -> None:
        trade_start_time, _ = self.trade_calendar.get_step_time()
        execute_result: list = []

        for order in self._get_order_iterator(event.payload.trade_decision):
            # execute the order.
            # NOTE: The trade_account will be changed in this function
            trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                order,
                trade_account=self.trade_account,
                dealt_order_amount=self.dealt_order_amount,
            )
            execute_result.append((order, trade_val, trade_cost, trade_price))
            self._update_dealt_order_amount(order)
            if self.verbose:
                print(
                    "[I {:%Y-%m-%d %H:%M:%S}]: {} {}, price {:.2f}, amount {}, deal_amount {}, factor {}, "
                    "value {:.2f}, cash {:.2f}.".format(
                        trade_start_time,
                        "sell" if order.direction == Order.SELL else "buy",
                        order.stock_id,
                        trade_price,
                        order.amount,
                        order.deal_amount,
                        order.factor,
                        trade_val,
                        self.trade_account.get_cash(),
                    ),
                )

        self._last_res = execute_result
        self._last_kwargs = {"trade_info": execute_result}

    def get_all_executors(self) -> List[BaseExecutorV2]:
        return [self]
