import gc
import torch


def get_module_size(module):
    num_params = sum(p.numel() for p in module.parameters())
    size_bytes = num_params * module.parameters().__next__().element_size()

    print(f"Number of parameters: {num_params:,}")
    print(f"Size in bytes: {size_bytes:,}")


def get_cuda_tensors():
    return [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda]


def move_list_to_device(liste, device):
    """ moves all tensors in the list to the given device

    :param liste: list of tensors
    :param device: device to move the tensors to
    """

    for i, o in enumerate(liste):
        if type(o) == list:
            move_list_to_device(o, device)
        else:
            if type(o) == torch.Tensor:
                liste[i] = o.to(device=device)


def move_dict_to_device(dictionary, device):
    """ moves all tensors in the dictionary to the given device

    :param dictionary: dictionary of tensors
    :param device: device to move the tensors to
    """

    for key, value in dictionary.items():
        if type(value) == dict:
            move_dict_to_device(value, device)
        else:
            if type(value) == torch.Tensor:
                dictionary[key] = value.to(device=device)


def concatenate_nested(structure1, structure2, dim=0):
    # Check if the input structures are tensors (leaves)
    if isinstance(structure1, torch.Tensor) and isinstance(structure2, torch.Tensor):
        # Concatenate the tensors along the appropriate dimension
        return torch.cat([structure1, structure2], dim=dim)

    # Check if the input structures have the same type (list or tuple)
    if type(structure1) == type(structure2):
        if isinstance(structure1, list):
            # If it's a list, recursively concatenate its elements
            return [concatenate_nested(elem1, elem2, dim=dim) for elem1, elem2 in zip(structure1, structure2)]
        elif isinstance(structure1, tuple):
            # If it's a tuple, recursively concatenate its elements
            return tuple(concatenate_nested(elem1, elem2, dim=dim) for elem1, elem2 in zip(structure1, structure2))

    # If the structures are not of the same type or are not tensors, return None or raise an error
    raise ValueError("Input structures are not compatible for concatenation.")


def slice_nested(structure, start_idx, end_idx):
    def slice_tensor(tensor, start_idx, end_idx):
        return tensor[start_idx:end_idx]

    if isinstance(structure, torch.Tensor):
        # slice tensors
        return slice_tensor(structure, start_idx, end_idx)
    
    if isinstance(structure, list):
        # recursively slice list elements
        return [slice_nested(elem, start_idx, end_idx) for elem in structure]

    if isinstance(structure, tuple):
        # recursively slice tuple elements
        return tuple(slice_nested(elem, start_idx, end_idx) for elem in structure)

    # If the structure is not a tensor, list, or tuple, return None or raise an error
    raise ValueError("Input structure is not supported for slicing.")

