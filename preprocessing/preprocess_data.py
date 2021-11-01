import utils.preprocessing_utils

def main():
	filename = input("Input xlsx file: ")
	df, id_to_covariates = utils.preprocessing_utils.import_excel(filename)
	tensor = utils.preprocessing_utils.convert_to_tensor(df)
	print(id_to_covariates)
	print("-----Get tensor time slice-----")
	print(utils.preprocessing_utils.get_tensor_time_slice(tensor, 1, 3, 0))
	print("-----Get tensor intervention slice-----")
	print(utils.preprocessing_utils.get_tensor_intervention_slice(tensor, 1, 0))
	print("-----Get tensor intervention slices-----")
	print(utils.preprocessing_utils.get_tensor_intervention_slices(tensor, 0))
	print("-----Get tensor time_intervention_slices-----")
	print(utils.preprocessing_utils.get_tensor_time_intervention_slices(tensor, 1, 3, 0))

if __name__ == "__main__":
    main()