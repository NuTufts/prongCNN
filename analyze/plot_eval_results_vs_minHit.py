
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("make minHit plots")
parser.add_argument("-e", "--epoch", type=int, default=10, help="use checkpoint for this epoch")
args = parser.parse_args()

minHitVals = [10, 30, 50, 70, 90]
#logfiles = ["prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_trex.log",
#            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_41113189.log",
#            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_41113190.log",
#            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_trex.log",
#            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_41113198.log"]
logfiles = ["prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_run4_lr1em6_41470831.log",
            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_run4_lr1em6_41466455.log",
            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_run4_lr1em6_41469169.log",
            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_run4_lr1em6_41469170.log",
            "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_run4_lr1em6_41469173.log"]

acc = []
acc_el = []
acc_ph = []
acc_mu = []
acc_pi = []
acc_pr = []

for logfile in logfiles:

  logpath = "/home/matthew/microboone/tufts/prongCNN/train/logs/minHitStudy/"+logfile
  log = open(logpath, "r")
  for line in log:
    if "EPOCH: %i"%args.epoch in line:
      data = line.split()
      break
  log.close()

  for i in range(len(data)-3):
    if data[i] == "test" and data[i+1] == "accuracy:" and data[i+3] == "electron":
      acc.append(float(data[i+2]))
    if data[i] == "electron" and data[i+1] == "test" and data[i+2] == "accuracy:":
      acc_el.append(float(data[i+3]))
    if data[i] == "photon" and data[i+1] == "test" and data[i+2] == "accuracy:":
      acc_ph.append(float(data[i+3]))
    if data[i] == "muon" and data[i+1] == "test" and data[i+2] == "accuracy:":
      acc_mu.append(float(data[i+3]))
    if data[i] == "pion" and data[i+1] == "test" and data[i+2] == "accuracy:":
      acc_pi.append(float(data[i+3]))
    if data[i] == "proton" and data[i+1] == "test" and data[i+2] == "accuracy:":
      acc_pr.append(float(data[i+3]))

fig = plt.figure(0)
plt.plot(minHitVals, acc, 'k-', label='overall')
plt.plot(minHitVals, acc_el, 'g-', label='electron')
plt.plot(minHitVals, acc_ph, 'y-', label='photon')
plt.plot(minHitVals, acc_mu, 'r-', label='muon')
plt.plot(minHitVals, acc_pi, 'b-', label='pion')
plt.plot(minHitVals, acc_pr, 'c-', linewidth=5, label='proton')
plt.title("Validation Accuracies After %i Epochs"%args.epoch)
plt.xlabel("minimum number of hits in training and validation images")
plt.legend()
fig.savefig("accuracy_vs_minHit_plots_%iepochs_prHighlight.png"%args.epoch)

