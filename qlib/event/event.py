from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, List, Optional

from qlib.backtest import BaseTradeDecision


def _connect(p: Optional[Event], q: Optional[Event]) -> None:
    if p is not None:
        p.next = q
    if q is not None:
        q.prev = p


@dataclass
class ExecutorPayload:
    trade_decision: BaseTradeDecision
    return_value: dict = None
    level: int = 0


class Event(object, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.prev: Optional[Event] = None
        self.next: Optional[Event] = None


class EmptyEvent(Event):
    def __init__(self) -> None:
        super(EmptyEvent, self).__init__()


class DecisionEvent(Event):
    def __init__(self, payload: object, event_type: object) -> None:
        super(DecisionEvent, self).__init__()
        self.payload = payload
        self.event_type = event_type


class CascadeEvent(Event):
    def __init__(
        self,
        payload: ExecutorPayload,
        fn: Callable[[CascadeEvent], None] = None,
    ) -> None:
        super(CascadeEvent, self).__init__()
        self.payload = payload
        self.fn = fn
        self.child_head: Optional[Event] = None
        self.child_tail: Optional[Event] = None

    def expand(self) -> None:
        if self.child_head is None:
            _connect(self.prev, self.next)
        else:
            _connect(self.prev, self.child_head)
            _connect(self.child_tail, self.next)

    def insert_event(self, event: Event) -> None:
        if self.child_head is None:
            self.child_head = self.child_tail = event
        else:
            _connect(event, self.child_head)
            self.child_head = event

    def insert_events(self, events: List[Event]) -> None:
        for event in events[::-1]:  # Last in first out
            self.insert_event(event)


class EventBuffer(object):
    def __init__(self) -> None:
        self._head = self._tail = EmptyEvent()

    def append(self, events: Event | List[Event]) -> None:
        event_list = events if isinstance(events, list) else [events]
        for event in event_list:
            _connect(self._tail, event)
            self._tail = event

    def head(self) -> Optional[Event]:
        return self._head.next
