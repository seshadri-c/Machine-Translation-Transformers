#Import Headers
import os
import wget
import shutil
import tarfile

def download_data_and_unzip(data_directory):
	
	dataset_name = "multi30k"
	data_path = os.path.join(data_directory,dataset_name)
	
	train_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz"
	test_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz"
	validation_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz"
	
	raw_data_path = os.path.join(data_path,"raw_data")
	uncompressed_data_path = os.path.join(data_path,"uncompressed_data")
		
	#shutil.rmtree(data_path, ignore_errors=True)
	if not os.path.exists(data_path):
			
		os.mkdir(data_path)
		os.mkdir(raw_data_path)
		os.mkdir(uncompressed_data_path)
		
		print("\nDownloading Training Data")
		wget.download(train_url,raw_data_path)
		print("\nDownloading Testing Data")
		wget.download(test_url,raw_data_path)
		print("\nDownloading Validation Data")
		wget.download(validation_url,raw_data_path)
		
		print("\nUncompressing the Data")
		file_names = os.listdir(raw_data_path)
		for f in file_names:
			my_tar = tarfile.open(os.path.join(raw_data_path,f))
			my_tar.extractall(uncompressed_data_path)
		print("Done..!!\n")
	
	return uncompressed_data_path
	
	
directory = "./data"
uncompressed_data_path = download_data_and_unzip(directory)

