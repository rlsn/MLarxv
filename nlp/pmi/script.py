from asgn3 import *

rank_353, sims_353 = compute_similarity(wordsim353, PPMI, cos_sim)
rank_353_soc, sims_353_soc = compute_similarity(wordsim353, PPMI, socpmi_sim, delta=6, gamma=3)
rank_353_pmi, sims_353_pmi = compute_similarity(wordsim353, PMI, cos_sim)
rank_353_jac, sims_353_jac = compute_similarity(wordsim353, PPMI, jaccard)

sims = [sims_353,sims_353_pmi,sims_353_jac,sims_353_soc]
labels = ["PPMI-COS", "PMI-COS", "PPMI-JAC", "SOCPMI"]
fig, ax = plt.subplots(2,4,figsize=(16,8))

for i,sim in enumerate(sims):
    minf,s1 = min_fre_sim(sim)
    maxf,s2 = max_fre_sim(sim)

    print("{0} min: pearson:{1:.3f},spearman:{2:.3f}".format(labels[i],pearson_cor(minf,s1),spearman_cor(minf,s1)))
    print("{0} max: pearson:{1:.3f},spearman:{2:.3f}".format(labels[i],pearson_cor(maxf,s2),spearman_cor(maxf,s2)))
    
    ax[0,i].plot(minf, s1, '.')
    ax[1,i].plot(maxf, s2, '.', color="C1")

    ax[0,i].set_xlabel("minimum frequency")
    ax[1,i].set_xlabel("maximum frequency")
    ax[0,i].set_ylabel("similarity")
    ax[1,i].set_ylabel("similarity")

    ax[0,i].set_xscale("log")
    ax[1,i].set_xscale("log")
    
    ax[0,i].set_title("{}\nsimilarity & minimum frequency".format(labels[i]))
    ax[1,i].set_title("{}\nsimilarity & maximum frequency".format(labels[i]))    

fig.tight_layout()
fig.savefig("sim-freq.png")