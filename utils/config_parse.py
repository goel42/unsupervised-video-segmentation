import yaml

with open("utils/config.yaml", "r") as stream:
	try:
		params = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)


#TODO: code for get_param here
#TODO: code for save_act_func here