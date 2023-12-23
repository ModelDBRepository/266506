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

def run_sim(avalues,fvalues,LEE,LEI,LIE,LII,data):
	nest.ResetKernel()


	startbuild = time.time()


	dt = 0.1    # the resolution in ms
	simtime = 1400.0  # Simulation time in ms
	delay = 1.5    # synaptic delay in ms


	g = 5.0  # ratio inhibitory weight/excitatory weight
	eta = 2.0  # external rate relative to threshold rate
	epsilon = 0.01  # connection probability


	#order = 6500
	#NE = 4 * order  # number of excitatory neurons
	#NI = 1 * order  # number of inhibitory neurons
	NE = int(data['nE'])  # number of excitatory neurons
	NI = int(data['nI'])  # number of inhibitory neurons
	N_neurons = NE + NI   # number of neurons in total
	N_rec = 150      # record from 50 neurons

	iperm = np.random.permutation(NI)
	eperm = np.random.permutation(NE)
	lt=len(LEE[0])
	SLEE=[[eperm[int(LEE[0][i])-1] for i in range(lt)],[eperm[int(LEE[1][i])-1] for i in range(lt)]]
	lt=len(LEI[0])
	SLEI=[[eperm[int(LEI[0][i])-1] for i in range(lt)],[iperm[int(LEI[1][i])-1] for i in range(lt)]]
	lt=len(LIE[0])
	SLIE=[[iperm[int(LIE[0][i])-1] for i in range(lt)],[eperm[int(LIE[1][i])-1] for i in range(lt)]]
	lt=len(LII[0])
	SLII=[[iperm[int(LII[0][i])-1] for i in range(lt)],[iperm[int(LII[1][i])-1] for i in range(lt)]]

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
	drive = nest.Create('ac_generator')

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

			astim=avalues[ida]
			fstim=fvalues[idf]

			espikes = nest.Create("spike_detector")
			ispikes = nest.Create("spike_detector")


			nest.SetStatus(espikes, [{"label": "brunel-py-ex",
						  "withtime": True,
						  "withgid": True,
						  "to_file": True}])

			nest.SetStatus(ispikes, [{"label": "brunel-py-in",
						  "withtime": True,
						  "withgid": True,
						  "to_file": True}])
			
			nest.Connect(nodes_ex[0:N_rec], espikes, syn_spec="excitatory")
			nest.Connect(nodes_in[0:N_rec], ispikes, syn_spec="excitatory")

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

			#nest.raster_plot.savefig("nest_rplot_s"+str(int(astim))+"_f"+str(int(fstim))+".png")
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
				pylab.suptitle('Markram et al. 2015 (AC stim. '+str(int(astim))+' pA at '+str(int(fstim))+' Hz)')
			else:
				pylab.suptitle('Markram et al. 2015')

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

			sio.savemat("output_s"+str(int(astim))+"_f"+str(int(fstim))+"_events.mat", {'mr_evse':evse, 'mr_tse':tse-Troot, 'mr_evsi':evsi, 'mr_tsi':tsi-Troot, 'mr_NE':NE, 'mr_NI':NI,'mr_N_rec':N_rec, 'mr_T':simtime})
			Troot=Troot+simtime


# loading connectivity
data=sio.loadmat('sim_Markram.mat')
LEE=data['LEE']
LII=data['LII']
LEI=data['LEI']
LIE=data['LIE']


avalues=[0.0]
fvalues=[35.0]

run_sim(avalues,fvalues,LEE,LEI,LIE,LII,data)


