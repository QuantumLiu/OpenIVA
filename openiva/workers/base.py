from threading import Thread, Event


class StoppableThread(Thread):
    def __init__(self) -> None:
        super().__init__()
        self._stop_event = Event()

    def stop(self):
        '''
        Stop the thread
        '''
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()
