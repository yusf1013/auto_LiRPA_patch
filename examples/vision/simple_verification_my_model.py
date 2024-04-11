"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm, PerturbationL0NormPatch
from auto_LiRPA.perturbations import PerturbationL0Norm
from auto_LiRPA.utils import Flatten
import torch_model as tm


def get_adjustment_matrix(prev_layer_neuron_number, ground_truth_idx):
    # Create an nxn matrix filled with zeros
    matrix = torch.zeros(prev_layer_neuron_number, prev_layer_neuron_number)

    # Fill the main diagonal with -1
    matrix.fill_diagonal_(-1)

    # Fill the mth column with 1
    matrix[:, ground_truth_idx] = 1

    # Remove the mth row
    matrix = torch.cat((matrix[:ground_truth_idx], matrix[ground_truth_idx + 1:]), dim=0)

    return matrix


def run_verification(model, image, ptb, true_label):
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    N = image.shape[0]

    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = lirpa_model(image)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()
    n_classes = pred.shape[1]
    print('Demonstration 1: Bound computation and comparisons of different methods.\n')
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    C = get_adjustment_matrix(n_classes, true_label.item()).unsqueeze(0)
    for method in ["backward (CROWN)"]:
        print('Bounding method:', method)

        start_time = time.time()
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C, return_A=True,
                                                    needed_A_dict=required_A)
        print("Finished in {:.3f}s".format(time.time() - start_time))

        lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
            A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
        upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
            A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']

        # n_classes = len(lb[0])
        for i in range(N):
            print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
            for j in range(len(lb[0])):
                indicator = '(ground-truth)' if j == true_label[i] else ''
                print('f_{j}(x_0): {l:8.3f} <= f_{true_label}(x_0+delta) - f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                    j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator, true_label=true_label.item()))
        print()

        return ptb.concretize_lower(image, lower_A.view(1, 9, 28 * 28), lower_bias)


def run_recursive_verification(model, image, ptb, true_label, eps):
    previous_result = ptb.dif
    res = run_verification(model=model, image=image, ptb=ptb, true_label=true_label)
    print(res)
    if len(res) == 0:
        return True
    elif previous_result.size() == res.size() and previous_result == res:
        return False

    num_rows = res.size(0)
    split_index = num_rows // 2
    first_half = res[:split_index].clone()
    second_half = res[split_index:].clone()

    ptb_first_half = PerturbationL0NormPatch(eps=eps, image=image)
    ptb_first_half.lower_values = image.clone()
    ptb_first_half.upper_values = image.clone()
    row_indices = first_half[:, 1]
    col_indices = first_half[:, 2]
    ptb_first_half.lower_values[0][row_indices, col_indices] = ptb.lower_values[0][row_indices, col_indices]
    ptb_first_half.upper_values[0][row_indices, col_indices] = ptb.upper_values[0][row_indices, col_indices]
    ptb_first_half.dif = first_half

    # Do the same with second_half
    ptb_second_half = PerturbationL0NormPatch(eps=eps, image=image)
    ptb_second_half.lower_values = image.clone()
    ptb_second_half.upper_values = image.clone()
    row_indices = second_half[:, 1]
    col_indices = second_half[:, 2]
    ptb_second_half.lower_values[0][row_indices, col_indices] = ptb.lower_values[0][row_indices, col_indices]
    ptb_second_half.upper_values[0][row_indices, col_indices] = ptb.upper_values[0][row_indices, col_indices]
    ptb_second_half.dif = second_half

    # res = run_verification(model=model, image=image, ptb=ptb_first_half, true_label=true_label)
    # print(res)
    first_result = run_recursive_verification(model, image, ptb_first_half, true_label, eps)
    if not first_result:
        return False
    second_result = run_recursive_verification(model, image, ptb_second_half, true_label, eps)
    return second_result


def main():
    # global model, N, image, true_label, lirpa_model, eps, norm, ptb
    ## Step 1: Define computational graph by implementing forward()
    # This simple model comes from https://github.com/locuslab/convex_adversarial
    loaded_model = tm.model
    loaded_model.load_state_dict(torch.load('pytorch_model.pth'))
    # loaded_model = tm.mnist_6_200()
    # loaded_model.load_state_dict(torch.load('mnist_6_200_nat.pth')['state_dict'][0])
    # loaded_model = tm.ModifiedModel(loaded_model, 6)
    # loaded_model = tm.Simple3NN()
    # loaded_model = tm.ModifiedModel(loaded_model, 2)
    loaded_model.eval()
    model = loaded_model
    ## Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=torchvision.transforms.ToTensor())
    # For illustration we only use 2 image from dataset
    N = 1
    image = test_data.data[:N].view(N, 1, 28, 28)[0]
    image = image.to(torch.float32) / 255.0
    # image = torch.tensor([[1.0, 2.0, 3.0]])
    # image = torch.tensor([[1/6.0, 2.0/6.0, 3.0/6.0]])
    # image = torch.tensor([[0.2, 0.3, 0.5]])
    # true_label = test_data.targets[:N]
    true_label = torch.argmax(loaded_model(image)).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()
    ## Step 3: wrap model with auto_LiRPA
    # The second parameter is for constructing the trace of the computational graph,
    # and its content is not important.
    print('Running on', image.device)
    eps = torch.tensor([2, 2])
    ## Step 4: Compute bounds using LiRPA given a perturbation
    # eps = 2
    # norm = 1
    # ptb = PerturbationLpNorm(norm=norm, eps=eps)
    # ptb = PerturbationL0Norm(eps=eps)
    ptb = PerturbationL0NormPatch(eps=eps, image=image)
    result = run_recursive_verification(model, image, ptb, true_label, eps)
    print(result)


main()



# print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.\n')
# # There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
# # of the output layer, with respect to the input layer (the image).
# required_A = defaultdict(set)
# required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])

# for method in [
#         'backward (CROWN)']:
#     print("Bounding method:", method)
#     if 'Optimized' in method:
#         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
#     lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], return_A=True, needed_A_dict=required_A)
#     lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
#     upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']


# for i in range(len(upper_A[0])):
#     print(f"\nUpper: {upper_A[0][i]} + {upper_bias[0][i]}")
#     print(f"Lower:{lower_A[0][i]} + {lower_bias[0][i]}")

# print("\nUpper:")
# print(f"{upper_A[0][0]} + {upper_bias[0][0]}")
# print(f"{upper_A[0][1]} + {upper_bias[0][1]}")
#
# print("\nLower:")
# print(f"{lower_A[0][0]} + {lower_bias[0][0]}")
# print(f"{lower_A[0][1]} + {lower_bias[0][1]}")


# print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
#     print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
#     print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
#     print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
#     print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
#     print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
#     print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
#     print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
#     print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')
#
## An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
# Here we compute the margin between groundtruth class and groundtruth class + 1.
# If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
# lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

# C = torch.zeros(size=(N, 1, n_classes), device=image.device)
# groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
# target_label = (groundtruth + 1) % n_classes
# C.scatter_(dim=2, index=groundtruth, value=1.0)
# C.scatter_(dim=2, index=target_label, value=-1.0)
#
# print('Demonstration 3: Computing bounds with a specification matrix.\n')
# print('Specification matrix:\n', C)
#
# for method in ['backward (CROWN)']:
#     print('Bounding method:', method)
#     if 'Optimized' in method:
#         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
#     lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
#     for i in range(N):
#         print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
#         print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
#             j=true_label[i], target= target_label[0][0][0], l=lb[i][0].item(), u=ub[i][0].item()))
#     print()


# for method in [
#         'backward (CROWN)', "CROWN",
#         'CROWN-Optimized (alpha-CROWN)', "forward", "forward+backward", 'IBP', 'IBP+backward (CROWN-IBP)']:
