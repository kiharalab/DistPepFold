def collate_fn(batch, args):
	collate_dict = {}
	collate_dict['test'] = batch[0]['test']
	return collate_dict, batch[0]['target']

