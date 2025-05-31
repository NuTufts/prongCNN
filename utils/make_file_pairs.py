import os,sys

lmreco_proddir="/cluster/tufts/wongjiradlabnu/twongj01/gen2/dlgen2prod/larmatch_and_reco_scripts/"

merged_dlreco_v = [
    "filelist_mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge.txt",
    "filelist_mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge.txt"
]

good_reco_lists = [
    "goodoutput_list_mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge_v3dev_reco_retune.txt",
    "goodoutput_list_mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge_v3dev_reco_retune.txt"
]

fout = open('filepairs.txt','w')

for igood,goodlist in enumerate(good_reco_lists):
    merged_dlreco = merged_dlreco_v[igood]
    merged_dlreco_path = lmreco_proddir+"/filelists/"+merged_dlreco

    goodlistpath = lmreco_proddir+"/goodoutput_lists/"+goodlist
    print("parse together: ")
    print("  ",merged_dlreco)
    print("  ",goodlist)
    
    fmerged_dlreco = open( merged_dlreco_path, 'r' )
    ll = fmerged_dlreco.readlines()

    merged_dlreco_list = {}
    ifileid = 0
    for l in ll:
        l = l.strip()
        merged_dlreco_list[ifileid] = l
        ifileid += 1


    fgoodlist = open(goodlistpath,'r')

    fileid_v = []    
    good_reco_filepairs = {}
    ll = fgoodlist.readlines()
    for  l in ll:
        l = l.strip()
        fileid = int(l.split()[0])
        recopath = l.split()[1]

        if fileid in merged_dlreco_list:
            good_reco_filepairs[fileid] = (merged_dlreco_list[fileid],recopath)
            fileid_v.append( fileid )

    for fileid in fileid_v:
        mdr,reco = good_reco_filepairs[fileid]
        print(f"{mdr} {reco}",file=fout)
        
fout.close()




