

import argparse
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

parser = argparse.ArgumentParser("make minHit plots")
parser.add_argument("-ef", "--firstEpoch", type=int, default=1, help="first epoch to plot")
parser.add_argument("-el", "--lastEpoch", type=int, default=20, help="last epoch to plot")
parser.add_argument("--smoothing", action="store_true", help="use lowess smoothing")
parser.add_argument("-s", "--smoothFrac", type=float, default=0.2, help="lowess smoothing fraction hyperparameter")
args = parser.parse_args()

epochs = np.array([i for i in range(args.firstEpoch,args.lastEpoch+1)])
nE = len(epochs)

class trainData():
  def __init__(self, log):
    self.logfile = log
    self.classAcc = np.zeros(nE)
    self.compRMSE = np.zeros(nE)
    self.purRMSE = np.zeros(nE) 

dataDict = {
  "v2me04_twoTask_stepLR":
  trainData("prongCNN_run3bOverlays_multiTask_plAll_2inChan_5ClassHard_minHit10_b32_stepLR_42004863.log"),
  "v2me05_twoTask_stepLR":
  trainData("prongCNN_run3bOverlays_multiTask_plAll_2inChan_5ClassHard_minHit10_b32_stepLR_v2me05_46375612.log"),
  "v2me05_tripleTask_stepLR":
  trainData("prongCNN_run3bOverlays_tripleTask_plAll_2inChan_5ClassHard_minHit10_b32_stepLR_v2me05_46375614.log"),
  "v2me05_tripleTask_cycleLR":
  trainData("prongCNN_run3bOverlays_tripleTask_plAll_2inChan_5ClassHard_minHit10_b64_oneCycleLR_v2me05_46375615.log"),
  "v2me05_tripleTask_cycleLR_noPCTrain":
  trainData("prongCNN_run3bOverlays_tripleTask_plAll_2inChan_5ClassHard_minHit10_b64_oneCycleLR_v2me05_noPCTrain_46375616.log"),
  "v2me05_tripleTask_cycleLR_noPCTrainOrVal":
  trainData("prongCNN_run3bOverlays_tripleTask_plAll_2inChan_5ClassHard_minHit10_b64_oneCycleLR_v2me05_noPCTrainOrVal_46375617.log")
}

for trainConfig in dataDict:

  logpath = "/home/matthew/microboone/tufts/prongCNN/train/logs/"+dataDict[trainConfig].logfile
  log = open(logpath, "r")

  for line in log:

    if "EPOCH:" in line:

      epoch = int(line[line.find("EPOCH: ")+7:line.find("  train total loss:")])
      if epoch < args.firstEpoch or epoch > args.lastEpoch:
        continue
      data = line.split()

      for i in range(len(data)-3):
        if data[i] == "test" and data[i+1] == "class" and data[i+2] == "accuracy:":
          dataDict[trainConfig].classAcc[epoch-args.firstEpoch] = float(data[i+3])
        if data[i] == "test" and data[i+1] == "comp" and data[i+2] == "RMSE:":
          dataDict[trainConfig].compRMSE[epoch-args.firstEpoch] = float(data[i+3])
        if data[i] == "test" and data[i+1] == "purity" and data[i+2] == "RMSE:":
          dataDict[trainConfig].purRMSE[epoch-args.firstEpoch] = float(data[i+3])

  log.close()



if args.smoothing:

  fig_reco = plt.figure(0)
  plt.plot(epochs, lowess(dataDict["v2me04_twoTask_stepLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='reco: v2_me_04')
  plt.plot(epochs, lowess(dataDict["v2me05_twoTask_stepLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='reco: v2_me_05')
  plt.title("Reco Version Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_reco.savefig("reco_comparison_particle_class_accuracy_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_reco = plt.figure(1)
  plt.plot(epochs, lowess(dataDict["v2me04_twoTask_stepLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'k-', label='reco: v2_me_04')
  plt.plot(epochs, lowess(dataDict["v2me05_twoTask_stepLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='reco: v2_me_05')
  plt.title("Reco Version Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_reco.savefig("reco_comparison_completeness_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_task = plt.figure(3)
  plt.plot(epochs, lowess(dataDict["v2me05_twoTask_stepLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='two tasks')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_stepLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='three tasks')
  plt.title("Task Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_task.savefig("task_comparison_particle_class_accuracy_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_task = plt.figure(4)
  plt.plot(epochs, lowess(dataDict["v2me05_twoTask_stepLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'b-', label='two tasks')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_stepLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='three tasks')
  plt.title("Task Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_task.savefig("task_comparison_completeness_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_trainParam = plt.figure(6)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_stepLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_particle_class_accuracy_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_trainParam = plt.figure(7)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_stepLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_completeness_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_trainParam = plt.figure(8)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_stepLR"].purRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].purRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("purity RMSE")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_purity_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_purityCut = plt.figure(9)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].classAcc,epochs,frac=args.smoothFrac,return_sorted=False), 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_particle_class_accuracy_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_purityCut = plt.figure(10)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].compRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_completeness_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)
  
  fig_purityCut = plt.figure(11)
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR"].purRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].purRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, lowess(dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].purRMSE,epochs,frac=args.smoothFrac,return_sorted=False), 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison (Smoothed, Lowess %.2f)"%args.smoothFrac)
  plt.xlabel("epoch")
  plt.ylabel("purity RMSE")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_purity_rmse_vs_epoch_smoothed%.2f.png"%args.smoothFrac)


else:

  fig_reco = plt.figure(0)
  plt.plot(epochs, dataDict["v2me04_twoTask_stepLR"].classAcc, 'k-', label='reco: v2_me_04')
  plt.plot(epochs, dataDict["v2me05_twoTask_stepLR"].classAcc, 'b-', label='reco: v2_me_05')
  plt.title("Reco Version Comparison")
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_reco.savefig("reco_comparison_particle_class_accuracy_vs_epoch.png")
  
  fig_reco = plt.figure(1)
  plt.plot(epochs, dataDict["v2me04_twoTask_stepLR"].compRMSE, 'k-', label='reco: v2_me_04')
  plt.plot(epochs, dataDict["v2me05_twoTask_stepLR"].compRMSE, 'b-', label='reco: v2_me_05')
  plt.title("Reco Version Comparison")
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_reco.savefig("reco_comparison_completeness_rmse_vs_epoch.png")
  
  fig_task = plt.figure(3)
  plt.plot(epochs, dataDict["v2me05_twoTask_stepLR"].classAcc, 'b-', label='two tasks')
  plt.plot(epochs, dataDict["v2me05_tripleTask_stepLR"].classAcc, 'r-', label='three tasks')
  plt.title("Task Comparison")
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_task.savefig("task_comparison_particle_class_accuracy_vs_epoch.png")
  
  fig_task = plt.figure(4)
  plt.plot(epochs, dataDict["v2me05_twoTask_stepLR"].compRMSE, 'b-', label='two tasks')
  plt.plot(epochs, dataDict["v2me05_tripleTask_stepLR"].compRMSE, 'r-', label='three tasks')
  plt.title("Task Comparison")
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_task.savefig("task_comparison_completeness_rmse_vs_epoch.png")
  
  fig_trainParam = plt.figure(6)
  plt.plot(epochs, dataDict["v2me05_tripleTask_stepLR"].classAcc, 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].classAcc, 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison")
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_particle_class_accuracy_vs_epoch.png")
  
  fig_trainParam = plt.figure(7)
  plt.plot(epochs, dataDict["v2me05_tripleTask_stepLR"].compRMSE, 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].compRMSE, 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison")
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_completeness_rmse_vs_epoch.png")
  
  fig_trainParam = plt.figure(8)
  plt.plot(epochs, dataDict["v2me05_tripleTask_stepLR"].purRMSE, 'r-', label='step learn rate, batch size 32')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].purRMSE, 'g-', label='one cycle learn rate, batch size 64')
  plt.title("Training Parameter Comparison")
  plt.xlabel("epoch")
  plt.ylabel("purity RMSE")
  plt.legend()
  plt.grid()
  fig_trainParam.savefig("trainParam_comparison_purity_rmse_vs_epoch.png")
  
  fig_purityCut = plt.figure(9)
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].classAcc, 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].classAcc, 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].classAcc, 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison")
  plt.xlabel("epoch")
  plt.ylabel("particle classification accuracy")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_particle_class_accuracy_vs_epoch.png")
  
  fig_purityCut = plt.figure(10)
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].compRMSE, 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].compRMSE, 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].compRMSE, 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison")
  plt.xlabel("epoch")
  plt.ylabel("completeness RMSE")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_completeness_rmse_vs_epoch.png")
  
  fig_purityCut = plt.figure(11)
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR"].purRMSE, 'g-', label='purity cut in training and validation samples')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrain"].purRMSE, 'c-', label='purity cut in validation sample only')
  plt.plot(epochs, dataDict["v2me05_tripleTask_cycleLR_noPCTrainOrVal"].purRMSE, 'm-', label='no purity cut')
  plt.title("Purity Cut Comparison")
  plt.xlabel("epoch")
  plt.ylabel("purity RMSE")
  plt.legend()
  plt.grid()
  fig_purityCut.savefig("purityCut_comparison_purity_rmse_vs_epoch.png")
