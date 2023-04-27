
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

parser = argparse.ArgumentParser("make minHit plots")
parser.add_argument("-e", "--epoch", type=int, default=10, help="use checkpoint for this epoch")
args = parser.parse_args()

logfiles = ["prongCNN_run3bOverlays_multiTask_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_trex.log","hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o01_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41571961.log","hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o1_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_trex.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o2_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_trex.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o3_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_trex.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o4_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567881.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o5_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567906.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o6_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567907.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o7_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567908.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o8_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567909.log", "hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o9_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41567910.log","hardTaskWeights/prongCNN_run3bOverlays_multiTask_hardWeight_partClass0o99_plAll_2inChan_5ClassHard_minHit10_b32_lr1em3_41571962.log"]

#partWeights = np.linspace(0.1,0.9,9)
partWeights = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])
compWeights = 1.0 - partWeights

partAcc = []
partLoss = []
compRMSE = []
compLoss = []

refPartAcc = -1.
refPartLoss = 99.
refCompRMSE = 99.
refCompLoss = 99.

for logfile in logfiles:

  logpath = "/home/matthew/microboone/tufts/prongCNN/train/logs/"+logfile

  log = open(logpath, "r")
  for line in log:
    if "EPOCH: %i"%args.epoch in line:
      data = line.split()
      break
  log.close()

  for i in range(len(data)-3):
    if data[i] == "test" and data[i+1] == "class" and data[i+2] == "loss:":
      pLoss = float(data[i+3])
    if data[i] == "test" and data[i+1] == "class" and data[i+2] == "accuracy:":
      pAcc = float(data[i+3])
    if data[i] == "test" and data[i+1] == "comp" and data[i+2] == "loss:":
      cLoss = float(data[i+3])
    if data[i] == "test" and data[i+1] == "comp" and data[i+2] == "RMSE:":
      cRMSE = float(data[i+3])

  if "hardWeight" in logfile:
    partLoss.append(pLoss)
    partAcc.append(pAcc)
    compLoss.append(cLoss)
    compRMSE.append(cRMSE)
  else:
    refPartLoss = pLoss
    refPartAcc = pAcc
    refCompLoss = cLoss
    refCompRMSE = cRMSE

def weightConv(x):
  return 1.0 - x

lineHW = Line2D([0],[0], color='k', linewidth=1, linestyle='-')
lineLW = Line2D([0],[0], color='k', linewidth=1, linestyle='--')
lines = [lineHW, lineLW]
labels = ['hard-coded weights', 'learned weights reference']

titleSize = 15
axFontSize = 10
tickFontSize = 8
axLabelSpace = 8

lFig, lAx1 = plt.subplots(constrained_layout=True)
lAx1.plot(partWeights, np.array(partLoss), 'r-')
lAx1.plot(partWeights, np.array([refPartLoss for i in range(len(partWeights))]), 'r--')
lAx1.set_xlabel('particle classification loss weight', fontsize=axFontSize, labelpad=axLabelSpace)
lAx1.set_ylabel('particle classification loss', color='r', fontsize=axFontSize, labelpad=axLabelSpace-1)
lAx1.tick_params(axis='y', labelcolor='r', labelsize=tickFontSize)
lAx1.tick_params(axis='x', labelsize=tickFontSize)
plt.title('Validation Task Loss Results After '+str(args.epoch)+' epochs', y=1.15, fontsize=titleSize)
seclAx1 = lAx1.secondary_xaxis('top', functions=(weightConv, weightConv))
seclAx1.set_xlabel('completeness regression loss weight', fontsize=axFontSize, labelpad=axLabelSpace)
seclAx1.tick_params(axis='x', labelsize=tickFontSize)
lAx2 = lAx1.twinx()
lAx2.plot(partWeights, np.array(compLoss), 'b-')
lAx2.plot(partWeights, np.array([refCompLoss for i in range(len(partWeights))]), 'b--')
lAx2.set_ylabel('completeness regression loss', color='b', fontsize=axFontSize, labelpad=axLabelSpace+1)
lAx2.tick_params(axis='y', labelcolor='b', labelsize=tickFontSize)
lAx2.tick_params(axis='x', labelsize=tickFontSize)
plt.legend(lines, labels)
lFig.savefig("prongCNN_loss_vs_taskWeights_epoch%i.png"%args.epoch)

pFig, pAx1 = plt.subplots(constrained_layout=True)
pAx1.plot(partWeights, np.array(partAcc), 'r-')
pAx1.plot(partWeights, np.array([refPartAcc for i in range(len(partWeights))]), 'r--')
pAx1.set_xlabel('particle classification loss weight', fontsize=axFontSize, labelpad=axLabelSpace)
pAx1.set_ylabel('particle classification accuracy', color='r', fontsize=axFontSize, labelpad=axLabelSpace-1)
pAx1.tick_params(axis='y', labelcolor='r', labelsize=tickFontSize)
pAx1.tick_params(axis='x', labelsize=tickFontSize)
plt.title('Validation Task Performance Results After '+str(args.epoch)+' epochs', y=1.15, fontsize=titleSize)
secpAx1 = pAx1.secondary_xaxis('top', functions=(weightConv, weightConv))
secpAx1.set_xlabel('completeness regression loss weight', fontsize=axFontSize, labelpad=axLabelSpace)
secpAx1.tick_params(axis='x', labelsize=tickFontSize)
pAx2 = pAx1.twinx()
pAx2.plot(partWeights, np.array(compRMSE), 'b-')
pAx2.plot(partWeights, np.array([refCompRMSE for i in range(len(partWeights))]), 'b--')
pAx2.set_ylabel('completeness regression RMSE', color='b', fontsize=axFontSize, labelpad=axLabelSpace+1)
pAx2.tick_params(axis='y', labelcolor='b', labelsize=tickFontSize)
pAx2.tick_params(axis='x', labelsize=tickFontSize)
plt.legend(lines, labels)
pFig.savefig("prongCNN_performance_vs_taskWeights_epoch%i.png"%args.epoch)

#plt.show()

