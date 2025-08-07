import os
import pickle
import argparse
import numpy as np
import scipy
import int_test

param_dir = None
header_dir = None

f_lit = lambda x: 'F_LIT(' + str(x) + ')'

def write_header(name, mats):
	contents = '#ifndef ' + name.upper() + '_H\n'
	contents += '#define ' + name.upper() + '_H\n'
	contents += '#include \'<libfixed/fixed.h>\'\n'
	contents += '#include \'<libdnn/mem.h>\'\n\n'
	for mat_name, mat, layer, sparse in mats:
		if layer == 'CONV' and sparse:
			mat_str = ''
			offsets_str = ''
			sizes_str = ''
			size = 0
			mat = mat.reshape(mat.shape[0], -1)
			for m in mat:
				data = m[m != 0.0].astype(dtype=str)

				idx = np.where(m != 0.0)[0]
				offsets = np.diff(idx).flatten()
				if data.shape[0] > 0:
					data_size = data.flatten().shape[0]
					str_mat = str(map(f_lit, data.flatten().tolist()))
					mat_str += str_mat.replace('[', '').replace(']', '') + ','

					str_offsets = str([idx[0]] + offsets.flatten().tolist())
					offsets_str += str_offsets.replace('[', '').replace(']', '') + ','

					sizes_str += str(data_size) + ','
					size += data_size
				else:
					sizes_str += '0,'

			mat_str = mat_str[:-1]
			offsets_str = offsets_str[:-1]
			sizes_str = sizes_str[:-1]
			layers = mat.shape[0]

			contents += '#define ' + mat_name.upper() + '_LEN ' + str(size) + '\n\n'

			contents += '__ro_hifram fixed ' + mat_name + \
				'[' + str(size) + '] = {' + mat_str + '};\n\n'

			contents += '__ro_hifram fixed ' + mat_name + '_offsets[' + \
				str(size) + '] = {' + offsets_str + '};\n\n'

			contents += '__ro_hifram fixed ' + mat_name + '_sizes[' + \
				str(layers) + '] = {' + sizes_str + '};\n\n'

		elif layer == 'FC' and sparse:
			csr = scipy.sparse.csr_matrix(mat)
			data, indices, indptr = csr.data, csr.indices, csr.indptr
			mat_str = str(map(f_lit, data.flatten().tolist()))
			mat_str = mat_str.replace('[', '{').replace(']', '}')
			indices_str = str(indices.flatten().tolist())
			indices_str = indices_str.replace('[', '{').replace(']', '}')
			indptr_str = str(indptr.flatten().tolist())
			indptr_str = indptr_str.replace('[', '{').replace(']', '}')

			contents += '#define ' + mat_name.upper() + '_LEN ' + \
				str(len(data)) + '\n\n'

			contents += '__ro_hifram fixed ' + mat_name + '[' + \
				str(len(data)) + '] = ' + mat_str + ';\n\n'

			contents += '__ro_hifram uint16_t ' + mat_name + '_offsets[' + \
				str(len(indices)) + '] = ' + indices_str + ';\n\n'

			contents += '__ro_hifram uint16_t ' + mat_name + '_sizes[' + \
				str(len(indptr)) + '] = ' + indptr_str + ';\n\n'
		else:
			mat_str = str(map(f_lit, mat.flatten().tolist()))
			mat_str = mat_str.replace('[', '{').replace(']', '}')
			shape_str = ''
			for s in mat.shape:
				shape_str += '[' + str(s) + ']'

			contents += '__ro_hifram fixed ' + mat_name + \
				shape_str + ' = ' + mat_str + ';\n\n'

	contents = contents.replace("'", '')
	contents += '#endif'
	path = os.path.join(header_dir, name + '.h')
	with open(path, 'w+') as f:
		f.write(contents)

def weight(name):
	global param_dir
	path = os.path.join(param_dir, name + '.param')
	with open(path, 'r') as f:
		data = pickle.load(f)
		print(path)
		print(data.shape)
		return data

def main(args):
	global header_dir, param_dir
	header_dir = args.header_dir
	param_dir = args.param_dir

	graph = int_test.Graph()

	graph.append('input', weight)
	graph.append('input_reshape', int_test.input_reshape, 'input')

	graph.append('conv1_w_', weight)
	graph.append('conv1_w', int_test.permute, 'conv1_w_')
	graph.append('conv1_b', weight)

	graph.append('conv2_w_', weight)
	graph.append('conv2_w', int_test.permute, 'conv2_w_')
	graph.append('conv2_b', weight)

	graph.append('fc1_w', weight)
	graph.append('fc1_wt', int_test.transpose, 'fc1_w')
	graph.append('fc1_b', weight)

	graph.append('fc2_w', weight)
	graph.append('fc2_wt', int_test.transpose, 'fc2_w')
	graph.append('fc2_b', weight)

	write_header('input', [
		('input', graph.eval('input_reshape'), 'FC', False)])

	write_header('conv1', [
		('conv1_w', graph.eval('conv1_w'), 'CONV', False), 
		('conv1_b', graph.eval('conv1_b'), 'FC', False)])

	write_header('conv2', [
		('conv2_w', graph.eval('conv2_w'), 'CONV', False), 
		('conv2_b', graph.eval('conv2_b'), 'FC', False)])

	write_header('fc1', [
		('fc1_w', graph.eval('fc1_wt'), 'FC', False), 
		('fc1_b', graph.eval('fc1_b'), 'FC', False)])
    
	write_header('fc2', [
		('fc2_w', graph.eval('fc2_wt'), 'FC', False), 
		('fc2_b', graph.eval('fc2_b'), 'FC', False)])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--param_dir',
		type=str,
		help='Parameter directory')
	parser.add_argument(
		'--header_dir',
		type=str,
		help='Header directory')
	args = parser.parse_args()
	main(args)