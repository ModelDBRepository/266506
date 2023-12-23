# source https://www.nest-simulator.org/py_sample/brunel_alpha_nest/

from scipy.optimize import fsolve

import nest
import nest.raster_plot
import numpy as np
import scipy.io as sio

import time
from numpy import exp


def LambertWm1(x): 
     return nest.ll_api.sli_func('LambertWm1', float(x)) 


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return (exp(1.0) / (tauSyn * CMem * b) *
            ((exp(-t_max / tauMem) - exp(-t_max / tauSyn)) / b -
             t_max * exp(-t_max / tauSyn)))

def run_sim(avalues,fvalues,LEE,LEI,LIE,LII):
	nest.ResetKernel()


	startbuild = time.time()


	dt = 0.1    # the resolution in ms
	simtime = 1400.0  # Simulation time in ms
	delay = 1.5    # synaptic delay in ms


	g = 5.0  # ratio inhibitory weight/excitatory weight
	eta = 2.0  # external rate relative to threshold rate
	epsilon = 0.01  # connection probability

	  
	N_neurons = NE + NI   # number of neurons in total
	N_rec = 150      # record from 50 neurons

	iperm = np.random.permutation(NI)
	eperm = np.random.permutation(NE)
	lt=len(LEE[0])
	SLEE=[[eperm[int(LEE[0][i])] for i in range(lt)],[eperm[int(LEE[1][i])] for i in range(lt)]]
	lt=len(LEI[0])
	SLEI=[[eperm[int(LEI[0][i])] for i in range(lt)],[iperm[int(LEI[1][i])] for i in range(lt)]]
	lt=len(LIE[0])
	SLIE=[[iperm[int(LIE[0][i])] for i in range(lt)],[eperm[int(LIE[1][i])] for i in range(lt)]]
	lt=len(LII[0])
	SLII=[[iperm[int(LII[0][i])] for i in range(lt)],[iperm[int(LII[1][i])] for i in range(lt)]]

	del LEE
	del LEI
	del LIE
	del LII

	CE = int(epsilon * NE)  # number of excitatory synapses per neuron
	CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
	C_tot = int(CI + CE)      # total number of synapses per neuron


	tauSyn = 0.5  # synaptic time constant in ms
	tauMem = 20.0  # time constant of membrane potential in ms
	CMem = 250.0  # capacitance of membrane in in pF
	theta = 20.0  # membrane threshold potential in mV
	neuron_params = {"C_m": CMem,
		         "tau_m": tauMem,
		         "tau_syn_ex": tauSyn,
		         "tau_syn_in": tauSyn,
		         "t_ref": 2.0,
		         "E_L": 0.0,
		         "V_reset": 0.0,
		         "V_m": 0.0,
		         "V_th": theta}
	J = 0.1        # postsynaptic amplitude in mV
	J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
	J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
	J_in = -g * J_ex    # amplitude of inhibitory postsynaptic current


	nu_th = (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
	nu_ex = eta * nu_th
	p_rate = 1000.0 * nu_ex * CE


	nest.SetKernelStatus({"resolution": dt, "print_time": True,
		              "overwrite_files": True})

	print("Building network")


	nest.SetDefaults("iaf_psc_alpha", neuron_params)
	nest.SetDefaults("poisson_generator", {"rate": p_rate})


	nodes_ex = nest.Create("iaf_psc_alpha", NE)
	nodes_in = nest.Create("iaf_psc_alpha", NI)
	noise = nest.Create("poisson_generator")

	# drive creation
	# driveparams = {'amplitude': astim, 'frequency': fstim}
	drive = nest.Create('ac_generator')
	# nest.SetStatus(drive, driveparams)

	print("Connecting devices")


	nest.CopyModel("static_synapse", "excitatory",
		       {"weight": J_ex, "delay": delay})
	nest.CopyModel("static_synapse", "inhibitory",
		       {"weight": J_in, "delay": delay})


	nest.Connect(noise, nodes_ex, syn_spec="excitatory")
	nest.Connect(noise, nodes_in, syn_spec="excitatory")


	nest.Connect(drive, nodes_ex, syn_spec="excitatory")
	nest.Connect(drive, nodes_in, syn_spec="excitatory")

	print("Connecting network")

	print("Excitatory connections")



	nest.Connect([nodes_ex[i] for i in SLEE[0]], [nodes_ex[i] for i in SLEE[1]], "one_to_one", "excitatory")
	nest.Connect([nodes_ex[i] for i in SLEI[0]], [nodes_in[i] for i in SLEI[1]], "one_to_one", "excitatory")

	print("Inhibitory connections")


	nest.Connect([nodes_in[i] for i in SLIE[0]], [nodes_ex[i] for i in SLIE[1]], "one_to_one", "inhibitory")
	nest.Connect([nodes_in[i] for i in SLII[0]], [nodes_in[i] for i in SLII[1]], "one_to_one", "inhibitory")


	endbuild = time.time()
	
	lf=len(fvalues)
	la=len(avalues)
	Troot=0
	for idf in range(0,lf):
		for ida in range(0,la):
			nest.ResetNetwork()
			
			espikes = nest.Create("spike_detector")
			ispikes = nest.Create("spike_detector")	

			nest.Connect(nodes_ex[0:N_rec], espikes, syn_spec="excitatory")
			nest.Connect(nodes_in[0:N_rec], ispikes, syn_spec="excitatory")	
			
			nest.SetStatus(espikes, [{"label": "brunel-py-ex",
		                  "withtime": True,
		                  "withgid": True,
		                  "to_file": True}])

			nest.SetStatus(ispikes, [{"label": "brunel-py-in",
		                  "withtime": True,
		                  "withgid": True,
		                  "to_file": True}])	
			
			astim=avalues[ida]
			fstim=fvalues[idf]

			driveparams = {'amplitude': astim, 'frequency': fstim}
			nest.SetStatus(drive, driveparams)

			print("Simulating")

			nest.Simulate(simtime)


			endsimulate = time.time()


			events_ex = nest.GetStatus(espikes, "n_events")[0]
			events_in = nest.GetStatus(ispikes, "n_events")[0]


			rate_ex = events_ex / simtime * 1000.0 / N_rec
			rate_in = events_in / simtime * 1000.0 / N_rec


			num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
					nest.GetDefaults("inhibitory")["num_connections"])


			build_time = endbuild - startbuild
			sim_time = endsimulate - endbuild


			print("Brunel network simulation (Python)")
			print("Number of neurons : {0}".format(N_neurons))
			print("Number of synapses: {0}".format(num_synapses))
			print("       Excitatory : {0}".format(int(CE * N_neurons) + N_neurons))
			print("       Inhibitory : {0}".format(int(CI * N_neurons)))
			print("Excitatory rate   : %.2f Hz" % rate_ex)
			print("Inhibitory rate   : %.2f Hz" % rate_in)
			print("Building time     : %.2f s" % build_time)
			print("Simulation time   : %.2f s" % sim_time)


			#nest.raster_plot.from_device(espikes, hist=True)

			#nest.raster_plot.savefig("nets_rplot_s"+str(int(astim))+"_f"+str(int(fstim))+".png")
			#nest.raster_plot.show()

			dSDe=nest.GetStatus(espikes, keys='events')[0]
			evse = dSDe["senders"]
			tse = dSDe["times"]

			dSDi=nest.GetStatus(ispikes, keys='events')[0]
			evsi = dSDi["senders"]
			tsi = dSDi["times"]


			import pylab

			pylab.figure(2)
			ms=2
			font = {'weight' : 'bold', 'size'   : 12}
			pylab.rc('font', **font)
			if astim>0:
				pylab.suptitle('Potjans and Diesmann 2012 (AC stim. '+str(int(astim))+' pA at '+str(int(fstim))+' Hz)')
			else:
				pylab.suptitle('Potjans and Diesmann 2012')

			pylab.subplot(121)
			pylab.plot(tse, evse, "|", color=[0,0,0], markersize=ms)
			pylab.title('Excitatory')
			pylab.xlabel('time (in ms)')
			pylab.ylabel('GID')
			pylab.xlim(0, simtime)
			pylab.ylim(0, N_rec)

			pylab.subplot(122)
			pylab.plot(tsi, evsi-NE, "|", color=[0,0,0], markersize=ms)
			pylab.title('Inhibitory')
			pylab.xlabel('time (in ms)')
			#pylab.ylabel('GID')
			pylab.xlim(0, simtime)
			pylab.ylim(0, N_rec)
			
			pylab.tight_layout(2.0)

			pylab.savefig("rplot_s"+str(int(astim))+"_f"+str(int(fstim))+".png")
			pylab.show()

			sio.savemat("output_s"+str(int(astim))+"_f"+str(int(fstim))+"_events.mat", {'ds_evse':evse, 'ds_tse':tse-Troot, 'ds_evsi':evsi, 'ds_tsi':tsi-Troot, 'ds_NE':NE, 'ds_NI':NI,'ds_N_rec':N_rec, 'ds_T':simtime})
			Troot=Troot+simtime

#Network creation

# Population size per layer
#          2/3e   2/3i   4e    4i    5e    5i    6e     6i    (Th not included) 
sf=2.4;
n_layer = [int(round(20683/sf)), int(round(5834/sf)), int(round(21915/sf)), int(round(5479/sf)), int(round(4850/sf)), int(round(1065/sf)), int(round(14395/sf)), int(round(2948/sf))] #, 902]

# Total cortical Population
N = sum(n_layer)

# Number of neurons accumulated
nn_cum = [0]
nn_cum.extend(np.cumsum(n_layer))

# Prob. connection table
table = np.array([[0.101,  0.169, 0.044, 0.082, 0.032, 0.,     0.008, 0.], #     0.    ],
               [0.135,  0.137, 0.032, 0.052, 0.075, 0.,     0.004, 0.], #     0.    ],
               [0.008,  0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.], #    0.0983],
               [0.069,  0.003, 0.079, 0.160, 0.003, 0.,     0.106, 0.], #    0.0619],
               [0.100,  0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.], #    0.    ],
               [0.055,  0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.], #    0.    ],
               [0.016,  0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225], #  0.0512],
               [0.036,  0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144]]) #,  0.0196]])

n_exc=[n_layer[0],n_layer[2],n_layer[4],n_layer[6]]
n_inh=[n_layer[1],n_layer[3],n_layer[5],n_layer[7]]

exc_cum=[0]
exc_cum.extend(np.cumsum(n_exc))
inh_cum=[0]
inh_cum.extend(np.cumsum(n_inh))

NE=sum(n_exc) # number of excitatory neurons
NI=sum(n_inh) # number of inhibitory neurons

LEEi=np.array([])
LEEj=np.array([])
LIEi=np.array([])
LIEj=np.array([])
LEIi=np.array([])
LEIj=np.array([])
LIIi=np.array([])
LIIj=np.array([])



for i in range(8):
 for j in range(8):
  p=table[j,i]
  lt=int(n_layer[i]*n_layer[j]*p)		
  if (i%2)==0:
   if (j%2)==0:
    LEEi=np.append(LEEi,exc_cum[int(i/2)]+np.random.randint(n_layer[i],size=lt))
    LEEj=np.append(LEEj,exc_cum[int(j/2)]+np.random.randint(n_layer[j],size=lt))
   else:
    LEIi=np.append(LEIi,exc_cum[int(i/2)]+np.random.randint(n_layer[i],size=lt))
    LEIj=np.append(LEIj,inh_cum[int((j-1)/2)]+np.random.randint(n_layer[j],size=lt))
  else:
   if (j%2)==0:
    LIEi=np.append(LIEi,inh_cum[int((i-1)/2)]+np.random.randint(n_layer[i],size=lt))
    LIEj=np.append(LIEj,exc_cum[int(j/2)]+np.random.randint(n_layer[j],size=lt))
   else:
    LIIi=np.append(LIIi,inh_cum[int((i-1)/2)]+np.random.randint(n_layer[i],size=lt))
    LIIj=np.append(LIIj,inh_cum[int((j-1)/2)]+np.random.randint(n_layer[j],size=lt))

# removing self loops
TLEEi=LEEi
TLEEj=LEEj
idx=np.where(TLEEi!=TLEEj)
LEEi=TLEEi[idx]
LEEj=TLEEj[idx]
del TLEEi
del TLEEj
del idx

TLIEi=LIEi
TLIEj=LIEj
idx=np.where(TLIEi!=TLIEj)
LIEi=TLIEi[idx]
LIEj=TLIEj[idx]
del TLIEi
del TLIEj

TLEIi=LEIi
TLEIj=LEIj
idx=np.where(TLEIi!=TLEIj)
LEIi=TLEIi[idx]
LEIj=TLEIj[idx]
del TLEIi
del TLEIj
del idx

TLIIi=LIIi
TLIIj=LIIj
idx=np.where(TLIIi!=TLIIj)
LIIi=TLIIi[idx]
LIIj=TLIIj[idx]
del TLIIi
del TLIIj
del idx


LEE=np.array([LEEi,LEEj])
LIE=np.array([LIEi,LIEj])
LEI=np.array([LEIi,LEIj])
LII=np.array([LIIi,LIIj])

print("Diesmann network Created")

avalues=[0.0]
fvalues=[35.0]


run_sim(avalues,fvalues,LEE,LEI,LIE,LII)




				


