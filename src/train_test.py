import torch

def box_normalization(y_tensor, b=[torch.tensor([110., 110.]), 220., 220.], inverse=False):
    b_c, b_w, b_h = b

    # Move b_c to the same device as y_tensor
    b_c = b_c.to(y_tensor.device)

    if inverse:
        matrix_inv = torch.tensor([[b_w, 0.],
                                [0., b_h]]).to(y_tensor.device) # Move matrix_inv to the same device as y_tensor
        y_tensor_matrix_applied = torch.matmul(y_tensor, matrix_inv.T)
        return y_tensor_matrix_applied + b_c

    else:
        matrix = torch.tensor([[1/b_w, 0.],
                           [0., 1/b_h]]).to(y_tensor.device) # Move matrix to the same device as y_tensor
        y_tensor_shifted = y_tensor - b_c
        return torch.matmul(y_tensor_shifted, matrix.T)


def l2_loss(tensor1, tensor2):
    """
    Computes the L2 loss between two (14, 2) tensors in a differentiable way,
    suitable for autograd and backpropagation.

    Parameters:
    tensor1 (torch.Tensor): A tensor of shape (14, 2).
    tensor2 (torch.Tensor): A tensor of shape (14, 2).

    Returns:
    torch.Tensor: A scalar tensor representing the summed L2 loss.
    """

    # Compute the element-wise squared differences
    diff = tensor1 - tensor2
    squared_diff = diff ** 2

    # Sum the squared differences along the row vectors (dim=1) to get per-row L2 values
    per_row_l2 = squared_diff.sum(dim=1).sqrt()

    # Sum all row-wise L2 distances to get the final loss
    loss = per_row_l2.sum()

    return loss

def calculate_pcp(pred_joints, gt_joints, threshold=0.5):
    """
    Calculate the Percentage of Correct Parts (PCP) given predicted and ground truth joint locations.
    
    Args:
        pred_joints (torch.Tensor): Predicted joints of shape (N, 14, 2).
        gt_joints (torch.Tensor): Ground truth joints of shape (N, 14, 2).
        threshold (float): Fraction of limb length for correct prediction.
        
    Returns:
        dict: PCP values for each limb.
    """

    # Define limb pairs based on joint indices
    limb_pairs = {
        "right_leg": [(0, 1), (1, 2)],         # Right ankle to knee, knee to hip
        "left_leg": [(5, 4), (4, 3)],          # Left ankle to knee, knee to hip
        "right_arm": [(6, 7), (7, 8)],         # Right wrist to elbow, elbow to shoulder
        "left_arm": [(11, 10), (10, 9)],       # Left wrist to elbow, elbow to shoulder
        "torso": [(2, 3), (2, 12), (3, 12)],   # Right hip to Left hip, hips to Neck
        "head_neck": [(12, 13)],               # Neck to Head top
    }
    
    # Initialize PCP counters
    pcp_results = {limb: 0 for limb in limb_pairs}
    total_counts = {limb: 0 for limb in limb_pairs}
    
    # Iterate over each sample
    for i in range(pred_joints.shape[0]):
        # Iterate over each limb
        for limb, pairs in limb_pairs.items():
            for (start, end) in pairs:
                # Ground truth limb length
                gt_limb_length = torch.norm(gt_joints[i, start] - gt_joints[i, end])
                
                # Predicted vs ground truth distance
                pred_distance = torch.norm(pred_joints[i, start] - gt_joints[i, start]) + \
                                torch.norm(pred_joints[i, end] - gt_joints[i, end])
                
                
                # Check if the predicted limb is within threshold of ground truth limb length
                if pred_distance / gt_limb_length <= threshold:
                    pcp_results[limb] += 1  # Correct prediction
                total_counts[limb] += 1    # Total predictions for this limb

    # Calculate PCP as a percentage for each limb
    pcp_percentages = {limb: (pcp_results[limb] / total_counts[limb]) * 100 
                       if total_counts[limb] > 0 else 0
                       for limb in limb_pairs}

    return pcp_percentages



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = l2_loss

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        targets = box_normalization(targets)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

            if args.dry_run:
                break



def test_pcp(args, model, device, test_loader, threshold = 0.5):
    model.eval()  # Set the model to evaluation mode

    # Initialize a dictionary to hold PCP scores for each limb
    pcp_scores = {
        "right_leg": 0.0,
        "left_leg": 0.0,
        "right_arm": 0.0,
        "left_arm": 0.0,
        "torso": 0.0,
        "head_neck": 0.0
    }
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)

            # Forward pass: get model predictions
            outputs = model(images).squeeze()
            outputs = box_normalization(outputs, inverse=True)
            

            # Calculate PCP scores
            batch_pcp_scores = calculate_pcp(outputs, targets, threshold)

            # Accumulate scores for averaging later
            for limb in pcp_scores.keys():
                pcp_scores[limb] += batch_pcp_scores[limb]

            if batch_idx % args.log_interval == 0:
                print('Test Batch: [{} / {} ({:.0f}%)]\tPCP Scores: {}'.format(
                    batch_idx * len(images), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), batch_pcp_scores))

            if args.dry_run:
                break

    # Average the scores over the number of batches
    num_batches = len(test_loader)
    for limb in pcp_scores.keys():
        pcp_scores[limb] /= num_batches

    print('Average PCP Scores:', pcp_scores)