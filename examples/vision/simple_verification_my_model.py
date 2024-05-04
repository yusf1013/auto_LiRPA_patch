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
    # print('Demonstration 1: Bound computation and comparisons of different methods.\n')
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    C = get_adjustment_matrix(n_classes, true_label.item()).unsqueeze(0)
    for method in ["backward (CROWN)"]:
        # print('Bounding method:', method)

        # start_time = time.time()
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C, return_A=True,
                                                    needed_A_dict=required_A)
        # print("Finished in {:.3f}s".format(time.time() - start_time))

        lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
            A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
        # upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
        #     A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']

        # # n_classes = len(lb[0])
        # for i in range(N):
        #     print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
        #     for j in range(len(lb[0])):
        #         indicator = '(ground-truth)' if j == true_label[i] else ''
        #         print('f_{j}(x_0): {l:8.3f} <= f_{true_label}(x_0+delta) - f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
        #             j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator, true_label=true_label.item()))
        # print()

        # return ptb.concretize_lower(image, lower_A.view(1, 9, 28 * 28), lower_bias)
        return ptb.split(image, lower_A.view(1, 9, 28 * 28), lower_bias)


def run_recursive_verification(model, image, ptb, true_label, eps, depth=0):


    # start_time = time.time()
    ptb1, ptb2 = run_verification(model=model, image=image, ptb=ptb, true_label=true_label)
    # print("Finished in {:.3f}s".format(time.time() - start_time))

    if ptb1.dif == 0 and ptb2.dif == 0:
        return True
    if ptb.dif == 1 and (ptb1.dif == 1 or ptb2.dif == 1):
        if torch.argmax(model(ptb1.get_average())) != true_label:
            return False
        elif torch.argmax(model(ptb1.lower_values)) != true_label:
            return False
        elif torch.argmax(model(ptb1.upper_values)) != true_label:
            return False
        else:

            return run_recursive_verification(model, ptb1.get_average(), ptb1, true_label, eps, depth=depth+1) and run_recursive_verification(model, ptb2.get_average(), ptb2, true_label, eps, depth=depth+1)
    return run_recursive_verification(model, image, ptb1, true_label, eps, depth=depth+1) and run_recursive_verification(model, image, ptb2, true_label, eps, depth=depth+1)





def main():
    safe = []
    unsafe = []
    # global model, N, image, true_label, lirpa_model, eps, norm, ptb
    ## Step 1: Define computational graph by implementing forward()
    # This simple model comes from https://github.com/locuslab/convex_adversarial
    loaded_model = tm.model
    loaded_model.load_state_dict(torch.load('pytorch_model.pth'))

    loaded_model = tm.mnist_6_200()
    loaded_model.load_state_dict(torch.load('mnist_6_200_nat.pth')['state_dict'][0])
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
    # N = 12
    # images = test_data.data[12:25].view(N, 1, 28, 28)
    images = test_data.data.unsqueeze(1)
    total_time = 0
    for i, image in enumerate(images[2:3]):
        print("Starting image", i)
        image = image.to(torch.float32) / 255.0

        true_label = torch.argmax(loaded_model(image)).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()

        print('Running on', image.device)
        eps = torch.tensor([2, 2])
        input_dim = torch.tensor(image[0].size())
        number_of_patches = input_dim - eps + torch.ones_like(eps)
        unlocked_patches = torch.ones(number_of_patches.tolist())

        ptb = PerturbationL0NormPatch(eps=eps, image=image, unlocked_patches=unlocked_patches)
        start_time = time.time()
        result = run_recursive_verification(model, image, ptb, true_label, eps)
        total_time += time.time() - start_time
        print("Finished FULL IMAGE in {:.3f}s".format(time.time() - start_time))
        print(result)
        if result:
            safe.append(i)
        else:
            unsafe.append(i)

    print(f"Safe: {safe}")
    print(f"Unsafe: {unsafe}")
    print(f"Total time: {total_time}")


if __name__ == '__main__':
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
