import threading
import time
from typing import Callable


class ProgressReporter:
    """
    Helper utility for reporting the state of a loop and the rate at which
    looping is occurring based on lapsed wall-time and a given reporting
    interval.

    Includes optional methods that are thread-safe.

    TODO: Add parameter for an optionally known total number of increments.
    """

    def __init__(
        self,
        log_func: Callable[[str], None],
        interval: float,
        what_per_second: str = "Loops"
    ):
        """
        Initialize this reporter.

        :param log_func: Logging function to use.
        :param interval: Time interval to perform reporting in seconds.  If no
            reporting during incrementation should occur, infinity should be
            passed.
        :param str what_per_second:
            String label about what is happening or being iterated over per
            second. The provided string should make sense when followed by
            " per second ...".

        """
        self.log_func = log_func
        self.interval = float(interval)
        self.what_per_second = what_per_second

        self.lock = threading.RLock()
        # c_last : Increment count at the time of the last report. Updated after
        #          report in ``increment_report``.
        # c      : Current Increment count, updated in ``increment_report``.
        # c_delta: Delta between the increment current and previous count at the
        #          time of the last report. Updated at the time of reporting in
        #          ``increment_report``.
        self.c_last = self.c = self.c_delta = 0
        # t_last : Time of the last report. Updated after report in
        #          ``increment_report``.
        # t      : Current time, Updated in ``increment_report``
        # t_delta: Delta between current time and the time of the last report.
        #          Updated in ``increment_report``.
        self.t_last = self.t = self.t_delta = self.t_start = 0.0

        self.started = False

    def start(self) -> "ProgressReporter":
        """ Start the timing state of this reporter.

        Repeated calls to this method resets the state of the reporting for
        multiple uses.

        This method is thread-safe.

        :returns: Self
        :rtype: ProgressReporter

        """
        with self.lock:
            self.started = True
            self.c_last = self.c = self.c_delta = 0
            self.t_last = self.t = self.t_start = time.time()
            self.t_delta = 0.0
        return self

    def increment_report(self) -> None:
        """
        Increment counter and time since last report, reporting if delta exceeds
        the set reporting interval period.
        """
        if not self.started:
            raise RuntimeError("Reporter needs to be started first.")
        self.c += 1
        self.c_delta = self.c - self.c_last
        self.t = time.time()
        self.t_delta = self.t - self.t_last
        # Only report if its been ``interval`` seconds since the last
        # report.
        if self.t_delta >= self.interval:
            self.report()
            self.t_last = self.t
            self.c_last = self.c

    def increment_report_threadsafe(self) -> None:
        """
        The same as ``increment_report`` but additionally acquires a lock on
        resources first for thread-safety.

        This version of the method is a little more costly due to the lock
        acquisition.
        """
        with self.lock:
            self.increment_report()

    def report(self) -> None:
        """
        Report the current state.

        Does nothing if no increments have occurred yet.
        """
        if not self.started:
            raise RuntimeError("Reporter needs to be started first.")
        # divide-by-zero safeguard
        if self.t_delta > 0 and (self.t - self.t_start) > 0:
            self.log_func("%s per second %f (avg %f) "
                          "(%d current interval / %d total)"
                          % (self.what_per_second,
                             self.c_delta / self.t_delta,
                             self.c / (self.t - self.t_start),
                             self.c_delta,
                             self.c))

    def report_threadsafe(self) -> None:
        """
        The same as ``report`` but additionally acquires a lock on
        resources first for thread-safety.

        This version of the method is a little more costly due to the lock
        acquisition.
        """
        with self.lock:
            self.report()
