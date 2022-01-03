import threading
import msvcrt


class Exit:
    def __init__(self):
        self.isExit = False
        self.thread = threading.Thread(target=self.work)

    def work(self):
        while True:
            newChar = msvcrt.getch()
            if newChar in b'\r':  # 如果是换行，则输入结束
                self.isExit = True
                break

    def run(self):
        self.thread.start()

    def get_status(self):
        return self.isExit

