from datetime import datetime

class logger(object):
    def __init__(self, logging_path):
        self.info_log = open(f"{logging_path}/info_log.txt", "a")
        self.info_log.flush()
        
        self.loss_log = open(f"{logging_path}/loss_log.csv", "a")
        self.loss_log.write('Epoch,Iteration,Loss_D,Loss_G,D_x,D_G_z1,D_G_z2\n')
        self.loss_log.flush()
    
    def info(self, msg):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f'[{curr_time}] {msg}'
        print(msg)
        self.info_log.write(msg + '\n')
        self.info_log.flush()
        
    def update_loss(self, msg):
        self.loss_log.write(msg)
        self.loss_log.flush()
        
    def close(self):
        self.loss_log.close()
        self.info_log.close()