
import argparse
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

parser = argparse.ArgumentParser("make minHit plots")
parser.add_argument("-ef", "--firstEpoch", type=int, default=1, help="first epoch to plot")
parser.add_argument("-el", "--lastEpoch", type=int, default=40, help="last epoch to plot")
parser.add_argument("--lowHitEval", action="store_true", help="plot from minHit 90-120 evaluation sample")
parser.add_argument("--smoothing", action="store_true", help="use lowess smoothing")
parser.add_argument("-s", "--smoothFrac", type=float, default=0.2, help="lowess smoothing fraction hyperparameter")
args = parser.parse_args()

epochs = np.array([i for i in range(args.firstEpoch,args.lastEpoch+1)])
nE = len(epochs)
acc = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]
acc_el = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]
acc_ph = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]
acc_mu = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]
acc_pi = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]
acc_pr = [np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE), np.zeros(nE)]

if args.lowHitEval:
  from logs.minHitStudy.minHit10to30_sample.logfile_list import logfiles
else:
  logfiles = ["prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_run1_lr1em3_combined.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_run2_lr1em4_41466454.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_run3_lr1em5_41469168.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit10_run4_lr1em6_41470831.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_run1_lr1em3_41343015.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_run2_lr1em4_41356843.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_run3_lr1em5_41384886.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit30_run4_lr1em6_41466455.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_run1_lr1em3_41312930.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_run2_lr1em4_41356844.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_run3_lr1em5_41384883.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit50_run4_lr1em6_41469169.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_run1_lr1em3_41343017.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_run2_lr1em4_41356845.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_run3_lr1em5_41384884.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit70_run4_lr1em6_41469170.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_run1_lr1em3_41343018.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_run2_lr1em4_41356846.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_run3_lr1em5_41384885.log", "prongCNN_8KPerClass_plAll_2inChan_5ClassHard_minHit90_run4_lr1em6_41469173.log"]


for logfile in logfiles:

  if args.lowHitEval:
    logpath = "/home/matthew/microboone/tufts/prongCNN/analyze/logs/minHitStudy/minHit10to30_sample/"+logfile
  else:
    logpath = "/home/matthew/microboone/tufts/prongCNN/train/logs/minHitStudy/"+logfile
  log = open(logpath, "r")

  if "minHit10" in logfile:
    idx = 0
  if "minHit30" in logfile:
    idx = 1
  if "minHit50" in logfile:
    idx = 2
  if "minHit70" in logfile:
    idx = 3
  if "minHit90" in logfile:
    idx = 4

  for line in log:

    if args.lowHitEval:
      dataString = "test accuracy:"
    else:
      dataString = "EPOCH:"

    if dataString in line:

      if args.lowHitEval:
        epoch = int(logfile.split("_")[9][5:])
      else:
        epoch = int(line[line.find("EPOCH: ")+7:line.find("  train loss:")])
      data = line.split()

      for i in range(len(data)-3):
        if data[i] == "test" and data[i+1] == "accuracy:" and data[i+3] == "electron":
          acc[idx][epoch-1] = float(data[i+2])
        if data[i] == "electron" and data[i+1] == "test" and data[i+2] == "accuracy:":
          acc_el[idx][epoch-1] = float(data[i+3])
        if data[i] == "photon" and data[i+1] == "test" and data[i+2] == "accuracy:":
          acc_ph[idx][epoch-1] = float(data[i+3])
        if data[i] == "muon" and data[i+1] == "test" and data[i+2] == "accuracy:":
          acc_mu[idx][epoch-1] = float(data[i+3])
        if data[i] == "pion" and data[i+1] == "test" and data[i+2] == "accuracy:":
          acc_pi[idx][epoch-1] = float(data[i+3])
        if data[i] == "proton" and data[i+1] == "test" and data[i+2] == "accuracy:":
          acc_pr[idx][epoch-1] = float(data[i+3])

  log.close()

if args.smoothing:

  fig = plt.figure(0)
  plt.plot(epochs, lowess(acc[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Overall Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig.savefig("accuracy_vs_epoch_overall_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))
  
  fig_el = plt.figure(1)
  plt.plot(epochs, lowess(acc_el[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc_el[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc_el[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc_el[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc_el[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Electron Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig_el.savefig("accuracy_vs_epoch_electron_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))
  
  fig_ph = plt.figure(2)
  plt.plot(epochs, lowess(acc_ph[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc_ph[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc_ph[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc_ph[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc_ph[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Photon Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig_ph.savefig("accuracy_vs_epoch_photon_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))
  
  fig_mu = plt.figure(3)
  plt.plot(epochs, lowess(acc_mu[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc_mu[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc_mu[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc_mu[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc_mu[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Muon Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig_mu.savefig("accuracy_vs_epoch_muon_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))
  
  fig_pi = plt.figure(4)
  plt.plot(epochs, lowess(acc_pi[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc_pi[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc_pi[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc_pi[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc_pi[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Pion Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig_pi.savefig("accuracy_vs_epoch_pion_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))
  
  fig_pr = plt.figure(5)
  plt.plot(epochs, lowess(acc_pr[0],epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='min hit threshold: 10')
  plt.plot(epochs, lowess(acc_pr[1],epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='min hit threshold: 30')
  plt.plot(epochs, lowess(acc_pr[2],epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='min hit threshold: 50')
  plt.plot(epochs, lowess(acc_pr[3],epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='min hit threshold: 70')
  plt.plot(epochs, lowess(acc_pr[4],epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='min hit threshold: 90')
  plt.title("Proton Accuracy (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.legend()
  fig_pr.savefig("accuracy_vs_epoch_proton_epochs%i-%i_smoothed%.2f.png"%(args.firstEpoch,args.lastEpoch,args.smoothFrac))


else:

  fig = plt.figure(0)
  plt.plot(epochs, acc[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc[4], 'c-', label='min hit threshold: 90')
  plt.title("Overall Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig.savefig("accuracy_vs_epoch_overall.png")
  
  fig_el = plt.figure(1)
  plt.plot(epochs, acc_el[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc_el[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc_el[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc_el[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc_el[4], 'c-', label='min hit threshold: 90')
  plt.title("Electron Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig_el.savefig("accuracy_vs_epoch_electron.png")
  
  fig_ph = plt.figure(2)
  plt.plot(epochs, acc_ph[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc_ph[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc_ph[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc_ph[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc_ph[4], 'c-', label='min hit threshold: 90')
  plt.title("Photon Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig_ph.savefig("accuracy_vs_epoch_photon.png")
  
  fig_mu = plt.figure(3)
  plt.plot(epochs, acc_mu[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc_mu[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc_mu[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc_mu[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc_mu[4], 'c-', label='min hit threshold: 90')
  plt.title("Muon Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig_mu.savefig("accuracy_vs_epoch_muon.png")
  
  fig_pi = plt.figure(4)
  plt.plot(epochs, acc_pi[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc_pi[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc_pi[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc_pi[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc_pi[4], 'c-', label='min hit threshold: 90')
  plt.title("Pion Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig_pi.savefig("accuracy_vs_epoch_pion.png")
  
  fig_pr = plt.figure(5)
  plt.plot(epochs, acc_pr[0], 'k-', label='min hit threshold: 10')
  plt.plot(epochs, acc_pr[1], 'g-', label='min hit threshold: 30')
  plt.plot(epochs, acc_pr[2], 'r-', label='min hit threshold: 50')
  plt.plot(epochs, acc_pr[3], 'b-', label='min hit threshold: 70')
  plt.plot(epochs, acc_pr[4], 'c-', label='min hit threshold: 90')
  plt.title("Proton Accuracy")
  plt.xlabel("epoch")
  plt.legend()
  fig_pr.savefig("accuracy_vs_epoch_proton.png")


