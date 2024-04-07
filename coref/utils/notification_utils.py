
import socket
import datetime
import requests
import traceback
import json

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SlackNotifier():

    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.channel = config.channel
        self.enable = config.enable
        self.webhook_url = config.webhook
        self.host_name = socket.gethostname()
        self.message_thread_id = ""
        self.dump = {
            "username": "Train-Notif",
            "channel": self.channel,
            "icon_emoji": ":clapper:",
        }

    def start_exp(self):
        if self.enable:
            self.start_time = datetime.datetime.now()
            dump = self.dump.copy()
            message = ['Training Started!',
                       'Experiment name: %s' % self.exp_name,
                       'Machine name: %s' % self.host_name,
                       'Starting date: %s' % self.start_time.strftime(DATE_FORMAT)]
            dump['text'] = '\n'.join(message)
            content = requests.post(self.webhook_url, json.dumps(dump))
            self.message_info = content
        # set the message id

    def log_info(self, info):
        """ Allow for generic updates on the thread (maybe per eval).
        """
        pass

    def exp_failed(self, ex):

        if self.enable:
            dump = self.dump.copy()
            end_time = datetime.datetime.now()
            elapsed_time = end_time - self.start_time
            self.start_time = datetime.datetime.now()
            dump = self.dump.copy()
            contents = ["Your training has crashed ☠️",
                        'Machine name: %s' % self.host_name,
                        'Experiment name: %s' % self.exp_name,
                        'Starting date: %s' % self.start_time.strftime(
                            DATE_FORMAT),
                        'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                        'Crashed training duration: %s\n\n' % str(
                            elapsed_time),
                        "Here's the error:",
                        '%s\n\n' % ex,
                        "Traceback:",
                        '%s' % traceback.format_exc()]
            dump['text'] = '\n'.join(contents)
            dump['icon_emoji'] = ':skull_and_crossbones:'
            content = requests.post(self.webhook_url, json.dumps(dump))
