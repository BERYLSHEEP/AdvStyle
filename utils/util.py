import torch
import numpy as np
import os

from boundaries.boundaries_setting import BOUNDARIES_POOL

'''
torch.Size([1, 512, 4, 4])
torch.Size([1, 512, 8, 8])
torch.Size([1, 512, 16, 16])
torch.Size([1, 512, 32, 32])
torch.Size([1, 256, 64, 64])
torch.Size([1, 128, 128, 128])
torch.Size([1, 64, 256, 256])
torch.Size([1, 32, 512, 512])
'''
_stylegan_network_shape = {4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64, 512:32}

class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean

def torch_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class noise_pool(object):
    def __init__(self, noise_num, dim, batch_size, is_save=True, save_path="", is_init=True, noise_dir=""):
        assert batch_size <= noise_num
        self.dim = dim
        self.noise_num = noise_num
        self.batch_size = batch_size
        self.noise_dir = noise_dir

        if is_init:
            self.noise_pool = torch.randn([noise_num, dim]).cuda()
            if is_save:
                noise_pool_np = torch_to_numpy(self.noise_pool)
                np.save(os.path.join(save_path, 'noise.npy'), noise_pool_np)
        else:
            self.noise_pool = self.__setup_files()
            if len(self.noise_pool) > self.noise_num:
                self.noise_pool = self.noise_pool[0:self.noise_num]

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        noise = []
        files_index_path = os.path.join(self.noise_dir, "files.npy")
        if os.path.exists(files_index_path):
            files_index = np.load(files_index_path)
            for index in files_index:
                possible_file = os.path.join(self.noise_dir, "{}_new.npy".format(index))
                noise.append(np.load(possible_file))
        else:
            files_index = []
            files_path = os.listdir(self.noise_dir)
            for name in files_path:
                name_index = name.split(".")[0].split("_")[0]
                files_index.append(int(name_index))
                possible_file = os.path.join(self.noise_dir, name)
                noise.append(np.load(possible_file))
            np.save(files_index_path, np.array(files_index))
            print(f"save files index in {files_index_path}")

        noise = torch.Tensor(noise).cuda()
        return noise

    def transform_noise(self, vector):
        assert vector.shape[1] == self.noise_num
        noise = vector.mm(self.noise_pool)
        return noise

    def get_noise(self, batch_size=None, value=1):
        # generate the noise index
        if batch_size != None:
            bs = batch_size
        else:
            bs = self.batch_size
        target_indices = torch.randint(0, self.noise_num, [bs], device='cuda')
        labels = self.one_hot(target_indices, value)
        noise = self.transform_noise(labels)
        return noise, target_indices

    def one_hot(self, index, value=1):
        '''
        Arguments:
            index {tensor} -- [the noise index]
        
        Keyword Arguments:
            value {int} -- [the noise scale] (default: {1})
        
        Returns:
            [tensor] -- [batch_size, self.noise_num]
        '''
        assert max(index) < self.noise_num 
        vec = torch.zeros([index.shape[0], self.noise_num])
        for b_index, target in enumerate(index):
            vec[b_index][target] = value 
        return vec.cuda()

def shift(z):
    # shitf the z with little movement
    scale = 0.3
    assert len(z.shape) == 2 and z.shape[1] == 512
    bs, dim = z.shape
    noise = torch.randn((bs, dim)).cuda()
    z = z + noise*scale
    return z

def make_noise(batch, dim):
    if isinstance(dim, int):
        dim = (dim,)
    return torch.randn((batch,) + dim)

def make_specific_noise(batch, dim, seed):
    rng = np.random.RandomState(seed)
    noise = torch.from_numpy(
        rng.standard_normal(dim * batch).reshape(batch, dim)).float() #[N, 512]
    return noise

def get_step_size(batch, train_alpha_b=1, train_alpha_a=-1):
    # step_sizes = (train_alpha_b - train_alpha_a) * \
    #     torch.randn((batch,)) + train_alpha_a  # sample step_sizes
    step_sizes = torch.Tensor(batch).uniform_(train_alpha_a, train_alpha_b).cuda()
    step_sizes = torch.clamp(step_sizes, min=0.3, max=train_alpha_b)*0.3
    return step_sizes

def make_direction(dim, number, drt_type, composing_type):
    if drt_type in ['z', 'w', 'w+']:
        if isinstance(number, int):
            drt_number = (number, )
        direction = torch.randn(drt_number + dim).cuda()
    elif drt_type == 'mid':
        dim = int(dim/2)
        direction = torch.randn([number, _stylegan_network_shape[dim], dim, dim]).cuda()

    if composing_type in ['z', 'w']:
        alpha = torch.full([number, 1], 1/number).cuda()
    elif composing_type == 'w+':
        alpha = torch.full([number, 1, 1], 0.1/number).cuda()
    elif composing_type == 'mid':
        alpha = torch.full([number, 1, 1, 1], 10/number).cuda()
    return [direction, alpha]

def get_w_start_end(attr_alias):
    start = []
    end = []
    for attr in attr_alias:
        start.append(BOUNDARIES_POOL[attr]['start'])
        end.append(BOUNDARIES_POOL[attr]['end'])
    return start,end
    
def manipulation(latent_codes, 
                boundary, 
                steps, 
                manipulation_layers=None,
                start_distance=-5.0,
                end_distance=5.0,
                num_layers=1,
                layerwise_manipulation=False,
                is_code_layerwise=False,
                is_boundary_layerwise=False):
    '''
    latent_codes: [num, *code_shape] or [num, num_layers, *code_shape]
    boundary: [1, *code_shape] or [[1,*code_shape],[1,*code_shape]]
    steps: [num]
    manipulation_layers: Indices of the layers to perform manipulation. (default: None)
    '''
    if isinstance(boundary, list):
        boundary_temp = boundary
    else:
        boundary_temp = [boundary]
    for temp in boundary_temp:
        if not (temp.ndim >= 2 and temp.shape[0] == 1):
            raise ValueError(f'Boundary should be with shape [1, *code_shape] or '
                         f'[1, num_layers, *code_shape], but '
                         f'{temp.shape} is received!')

    if not layerwise_manipulation:
        assert not is_code_layerwise
        assert not is_boundary_layerwise
        num_layers = 1
        manipulate_layers = None
        layerwise_manipulation_strength = 1.0

    # Make latent codes layer-wise if needed.
    assert num_layers > 0
    if not is_code_layerwise:
        x = latent_codes[:, np.newaxis]
        x = np.tile(x, [num_layers if axis == 1 else 1 for axis in range(x.ndim)])
    else:
        x = latent_codes
    
    if x.shape[1] != num_layers:
        raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')

    num = x.shape[0]
    code_shape = x.shape[2:]
    x = x[:, np.newaxis]
    # x.ndim : number of array dimensions
    results = np.tile(x, [steps if axis == 1 else 1 for axis in range(x.ndim)])

    if not isinstance(boundary, list):
        # manipulate several boundaries at the same time
        boundary = [boundary]
        manipulation_layers = [manipulation_layers]
        start_distance = [start_distance]
        end_distance = [end_distance]
    else:
        start_distance = start_distance.tolist()
        end_distance = end_distance.tolist()

    zeros_boundary = np.zeros((num, steps, num_layers, *code_shape), dtype=float)
    result_boundary = np.zeros((num, steps, num_layers, *code_shape), dtype=float)

    for bdy, layers, start, end in zip(boundary, manipulation_layers, start_distance, end_distance):
    # Preprocessing for layer-wise manipulation.
    # Parse indices of manipulation layers.
        layer_indices = parse_indices(
          layers, min_val=0, max_val=num_layers - 1)
        if not layer_indices:
            layer_indices = list(range(num_layers))    

    # Make boundary layer-wise if needed.
        if not is_boundary_layerwise:
            b = bdy
            b = np.tile(b, [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
        else:
            b = bdy[0]
        if b.shape[0] != num_layers:
            raise ValueError(f'Boundary should be with shape [num_layers, '
                           f'*code_shape], where `num_layers` equals to '
                           f'{num_layers}, but {b.shape} is received!')
        
        if x.shape[2:] != b.shape:
            raise ValueError(f'Latent code shape {x.shape} and boundary shape '
                         f'{b.shape} mismatch!')

        b = b[np.newaxis, np.newaxis, :]
        if steps == 1:
            l = np.array([end])
        else:
            if steps%2 == 1:
                per_step = int((steps - 1)/2)
            else:
                per_step = int(steps/2)
            if start != 0 and end != 0:
                l_left = np.linspace(start, 0, per_step, endpoint = False)
                l_right = np.linspace(0, end, per_step + 1)
            elif start == 0:
                l_left = []
                l_right = np.linspace(0, end, steps)
            elif end == 0:
                l_left = np.linspace(start, 0, steps)
                l_right = []
            l = np.concatenate((l_left, l_right))
        l = l.reshape([steps if axis == 1 else 1 for axis in range(x.ndim)])

        is_manipulatable = np.zeros(results.shape, dtype=bool)
        is_manipulatable[:, :, layer_indices] = True
        temp_result = np.where(is_manipulatable, l*b, zeros_boundary)
        result_boundary += temp_result

    # normalization
    if is_code_layerwise:
        latent_norm = np.linalg.norm(latent_codes, axis = 2) # [num, num_layers *code_shape]
    else:
        latent_norm = np.linalg.norm(latent_codes[:,np.newaxis], axis=2)
    scale = (latent_norm / np.linalg.norm(x + result_boundary, axis = 3))  
    scale = scale.reshape(scale.shape[0], scale.shape[1], scale.shape[2], 1) 
    scale_tile = np.tile(scale, [512 if axis == 3 else 1 for axis in range(scale.ndim)])
    results = scale_tile * (x + result_boundary)
    assert results.shape == (num, steps, num_layers, *code_shape)


    return results if layerwise_manipulation else results[:, :, 0]

def find_close_fast(arr, e):
    low = 0
    high = len(arr) - 1
    idx = -1
 
    while low <= high:
        mid = int((low + high) / 2)
        if e == arr[mid] or mid == low:
            idx = mid
            break
        elif e > arr[mid]:
            low = mid
        elif e < arr[mid]:
            high = mid
 
    if idx + 1 < len(arr) and abs(e - arr[idx]) > abs(e - arr[idx + 1]):
        idx += 1

    return idx

def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    If the input is a list or tuple, this function has no effect.

    The input can also be a string, which is either a comma separated list of
    numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
    be ignored.

    Args:
    obj: The input object to parse indices from.
    min_val: If not `None`, this function will check that all indices are equal
      to or larger than this value. (default: None)
    max_val: If not `None`, this function will check that all indices are equal
      to or smaller than this field. (default: None)

    Returns:
    A list of integers.

    Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
    else:
        raise ValueError(f'Invalid type of input: {type(obj)}!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

