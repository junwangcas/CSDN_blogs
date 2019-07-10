import numpy as np
import pickle
class nn_log(object):
    w = []
    b = []
    ws = []
    bs = []
    forward = []
    forwards = []
    file_w = open("data/w.txt",'wb')
    file_b = open("data/b.txt",'wb')
    file_forward = open("data/forward.txt",'wb')
    file_activation = open("data/activation.txt",'w+')
    file_partial = open("data/partial.txt",'w+')
    file_cost = open("data/cost.txt",'w+')

    def write_log(self):
        self.ws.append(self.w)
        self.bs.append(self.b)
        self.forwards.append(self.forward)
        # for w in self.w:
        #     np.savetxt(self.file_w,w,delimiter=",",)
        #     #self.file_w.write(np.array2string(w))
        #     #self.file_w.write(np.array2string(w[1,:]))
        #     #self.file_w.write("\n")
        # for b in self.b:
        #     np.savetxt("data/b_txt", b, delimiter=",")
        #     #np.savetxt("data/b_txt", "\n")
        #     #self.file_b.write(np.array2string(b))
        #     #self.file_b.write("\n")

    def close_file(self):
        pickle.dump(self.ws, self.file_w, 0)
        pickle.dump(self.bs, self.file_b, 0)
        pickle.dump(self.forwards, self.file_forward, 0)
        # for w in self.ws:
        #     for w_1 in w:
        #         np.savetxt(self.file_w, w_1, delimiter=",")
        # for b in self.bs:
        #     for b_1 in b:
        #         np.savetxt(self.file_b, b_1, delimiter=",")
        #print(0)
        self.file_w.close()
        self.file_b.close()
        self.file_forward.close()

nn_log_instance = nn_log()