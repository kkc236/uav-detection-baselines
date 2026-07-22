import torch

from src.ebc_qp_loss import P2Targets, compute_ebc_loss, compute_sparse_p2_loss


def test_sparse_p2_keeps_top50_union_positive_and_excludes_inside_gt_negatives():
    logits = torch.zeros(1, 6, 2, requires_grad=True)
    boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1]] * 6], requires_grad=True)
    targets = P2Targets(
        gt_boxes=[torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
        gt_classes=[torch.tensor([1])],
        assigned_pairs=[torch.tensor([[0, 4]])],
        topk_indices=torch.tensor([[0, 1, 2]]),
        anchor_centers=torch.tensor(
            [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7]]
        ),
    )

    result = compute_sparse_p2_loss(logits, boxes, targets)

    assert result.classification_indices[0].tolist() == [0, 2, 4]


def test_vfl_iou_target_is_detached_but_box_losses_train_boxes():
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], requires_grad=True)
    targets = P2Targets(
        gt_boxes=[torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
        gt_classes=[torch.tensor([1])],
        assigned_pairs=[torch.tensor([[0, 0]])],
        topk_indices=torch.tensor([[0]]),
        anchor_centers=torch.tensor([[0.5, 0.5]]),
    )

    result = compute_sparse_p2_loss(logits, boxes, targets)
    result.total.backward()

    assert result.vfl_target.requires_grad is False
    assert boxes.grad is not None
    assert torch.count_nonzero(boxes.grad) > 0


def test_no_positive_loss_is_finite_and_differentiable():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    boxes = torch.zeros(1, 3, 4, requires_grad=True)
    targets = P2Targets(
        gt_boxes=[torch.empty(0, 4)],
        gt_classes=[torch.empty(0, dtype=torch.long)],
        assigned_pairs=[torch.empty(0, 2, dtype=torch.long)],
        topk_indices=torch.tensor([[0, 1]]),
        anchor_centers=torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]),
    )

    result = compute_sparse_p2_loss(logits, boxes, targets)
    result.total.backward()

    assert torch.isfinite(result.total)
    assert boxes.grad is not None
    assert torch.count_nonzero(boxes.grad) == 0


def test_ebc_uses_correct_class_logit_only_for_uncovered_assigned_gt():
    logits = torch.tensor([[[-2.0, 0.4], [3.0, -1.0]]], requires_grad=True)

    loss = compute_ebc_loss(
        p2_logits=logits,
        assigned_pairs=[torch.tensor([[0, 0], [1, 1]])],
        gt_classes=[torch.tensor([1, 0])],
        uncovered=[torch.tensor([True, False])],
        stock_boundary=torch.tensor([1.0]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.6))


def test_ebc_without_eligible_targets_returns_differentiable_zero():
    logits = torch.zeros(1, 2, 2, requires_grad=True)

    loss = compute_ebc_loss(
        p2_logits=logits,
        assigned_pairs=[torch.tensor([[0, 0]])],
        gt_classes=[torch.tensor([1])],
        uncovered=[torch.tensor([False])],
        stock_boundary=torch.tensor([1.0]),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
