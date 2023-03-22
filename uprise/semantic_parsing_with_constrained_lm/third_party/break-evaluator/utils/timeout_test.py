
from time import sleep

from utils.timeout import exit_after


@exit_after(5)
def countdown(n):
    print('countdown started', flush=True)
    for i in range(n, -1, -1):
        print(i, end=', ', flush=True)
        sleep(1)
    print('countdown finished')


if __name__ == "__main__":
    try:
        countdown(10)
    except KeyboardInterrupt:
        print('timeout!')
