import matplotlib.pyplot as plt

def generate_figures(results_dict, results_path, partition, labels):
	print("Results")
	counter = 0
	plt.figure(num=1,figsize=[10,10])
	for key in results_dict.keys():
		plt.scatter(counter,results_dict[key]["MPCA"])
		plt.text(counter,results_dict[key]["MPCA"],key)
		counter+=1
	# plt.show();
	plt.savefig(results_path+"/MPCA.pdf",dpi=300)
	plt.close(1)
	counter = 0
	plt.figure(num=1,figsize=[10,10])
	for key in results_dict.keys():
		plt.scatter(results_dict[key]["Time"],results_dict[key]["MPCA"])
		plt.text(results_dict[key]["Time"],results_dict[key]["MPCA"],key)
		counter+=1
	# plt.show();
	plt.savefig(results_path+"/time.pdf",dpi=300)
	plt.close(1)
	return None