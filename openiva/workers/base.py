from threading import Thread, Event

# Deprecated
# class StoppableThread(Thread):
#     def __init__(self) -> None:
#         super().__init__()
#         self._stop_event = Event()

#     def stop(self):
#         '''
#         Stop the thread
#         '''
#         self._stop_event.set()

#     @property
#     def stopped(self):
#         return self._stop_event.is_set()

class StoppableThread(Thread):
    '''
    Base class for stoppable thread.
    Can be sttopped by calling stop() method.
    Can be interrupted by Ctrl+C
    '''
    def __init__(self) -> None:
        super().__init__()
        self.setDaemon(True)
        self._stop_event = Event()

    def stop(self):
        '''
        Stop the thread
        '''
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()
    
    def run(self):
        while not self.stopped:
            try:
                self._run()
            except KeyboardInterrupt:
                return
            except:
                continue

    def _run(self):
        raise NotImplementedError