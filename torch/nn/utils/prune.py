r"""
Pruning methods
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import torch

# TODO: should the pruning method "own" name? or a module? it doesn't
# make much sense for it to know the name of the tensor it acts on but
# not what module it's in. I think it has to know the name because 
# hooks in a module don't have any other way of knowing which tensor
# they refer to. Having the pruning method know about its module might
# create problems for serialization but it makes more logical sense to me.

class BasePruningMethod(ABC):

    def __init__(self):
        pass

    def __call__(self, module, inputs):
        """Multiplies the mask (stored in `module[name + '_mask']`)
        into the original tensor (stored in `module[name + '_orig']`)
        and stored the result into `module[name]` by using `apply_mask`.
        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))


    @abstractmethod
    def compute_mask(self, t):
        """Computes and returns a mask for the input thensor `t`.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            mask (torch.Tensor): mask to apply to `t`, of same dims as `t`
        """
        pass

    # @abstractmethod
    def apply_mask(self, module):
        """Simply handles the multiplication.
        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.
        Args:
            module (nn.Module): module containing the tensor to prune
        Returns:
            output (torch.Tensor): pruned tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        assert self._tensor_name is not None, "Module {} has to be pruned".format(
            module)  # this gets set in apply()
        mask = getattr(module, self._tensor_name + '_mask')
        orig = getattr(module, self._tensor_name + '_orig')
        output = mask.to(dtype=orig.dtype) * orig   
        return output

    @classmethod
    def apply(cls, module, name, *args, **kwargs):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            args: arguments passed on to a subclass of BasePruningMethod
            kwargs: keyword arguments passed on to a subclass of a BasePruningMethod
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            # Check if a pruning method has already been applied to
            # `module[name]`. If so, store that in `old_method`.
            old_method = None
            found = 0
            # there should technically be only 1 hook with hook.name == name
            # assert this using `found`
            for k, hook in module._forward_pre_hooks.items():
                # if it exists, take existing thing, remove hook, then 
                # go thru normal thing
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    # reset the tensor reparametrization
                    module = remove_pruning(module, name)
                    found += 1
            assert found <= 1, "Avoid adding multiple pruning hooks to the\
                same tensor {} of module {}. Use a PruningContainer.".format(
                    name, module)

            # Apply the new pruning method, either from scratch or on top of 
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            setattr(method, '_tensor_name', name)

            # combine `methods` with `old_method`, if `old_method` exists
            if old_method is not None:  # meaning that there was a hook
                # if the hook is already a pruning container, just add the
                # new pruning method to the container
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method  # rename old_method --> method

                # if the hook is simply a single pruning method, create a 
                # container, add the old pruning method and the new one
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    # Have the pruning method remember the name of its tensor
                    setattr(container, '_tensor_name', name)
                    container.add_pruning_method(method)
                    method = container  # rename container --> method
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)

        # Apply pruning to the module

        # original tensor, prior to this iteration of pruning
        orig = getattr(module, name)
        # copy `module[name]` to `module[name + '_orig']`
        module.register_parameter(name + '_orig', torch.nn.Parameter(orig.data))
        # temporarily delete `module[name]`
        del module._parameters[name]
        # get the final mask, computed according to the specific method
        mask = method.compute_mask(orig)
        # reparametrize by saving mask to `module[name + '_mask']`...
        module.register_buffer(name + '_mask', mask)
        # ... and the new pruned tensor to `module[name]`
        setattr(module, name, method.apply_mask(module))

        # associate the pruning method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)

        return method

    # @abstractmethod
    # def remove(self):
        # """Removes this pruning method from the forward hook. You can only
        # remove the very last pruning object from the history. Intermediate
        # pruning steps cannot be popped, as they would change all of the subsequent
        # masks and amounts.
        # Forward pass should stay the same as with the hook, but now all units
        # pruned in this iteration go back to being trainable in the backward pass.
        # """
    # TODO: should this also be a class method?
    def remove(self, module):
        r"""Removes the pruning reparameterization from a module. The pruned
        parameter named `name` remains permanently pruned, and the parameter
        named `name+'_orig'` is removed from the parameter list. Similarly,
        the buffer named `name+'_mask' is removed from the buffers.

        Note: 
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        assert self._tensor_name is not None, "Module {} has to be pruned\
            before pruning can be removed".format(module)  # this gets set in apply()

        # to register the op
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        delattr(module, self._tensor_name)
        del module._parameters[self._tensor_name + '_orig']
        # TODO: do I delete the buffer too?
        del module._buffers[self._tensor_name + '_mask']
        module.register_parameter(self._tensor_name, torch.nn.Parameter(weight.data))


class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods.
    """
    def __init__(self, *args):
        self._pruning_methods = tuple()  # is this the right structure here?
        # TODO: fill these
        # self.cumulative_masks = tuple()
        self.cumulative_amounts = tuple()

        if not isinstance(args, Iterable):  # only 1 item
            self.add_pruning_method(args)
        else:  # manual construction from list or other iterable
            for method in args:
                self.add_pruning_method(method)

    @classmethod
    def build_from(cls, obj):
        """Copy constructor
        Args:
            obj (PruningContainer): another PruningContainer object
        """
        if isinstance(obj, PruningContainer):
            # return copy.deepcopy(obj)
            return cls(obj.name, *obj._pruning_methods)
        else:
            raise TypeError("obj must be of type PruningContainer")

    def add_pruning_method(self, method):
        r"""Adds a child pruning method to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError("{} is not a BasePruningMethod subclass".format(
                type(method)))
        # check method has the same module and name as self
        # elif self.module != method.module:
        #     raise ValueError("Can only add pruning methods acting on"
        #         "parameters of module {} to PruningContainer {}.".format(
        #         self.module, self) + "Found {}".format(method.module))
        elif self._tensor_name != method._tensor_name:
            raise ValueError("Can only add pruning methods acting on"
                "the parameter named {} to PruningContainer {}.".format(
                name, self) + "Found {}".format(method.name))
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method, )

        # fill attributes
        self.cumulative_amounts += (self.get_cumulative_amount(), )

    def __len__(self):
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    # the following is inspired by torch.nn.modules.container
    def __getitem__(self, idx):
        """
        Args:
            idx (int or slice): 
        Returns:
            a BasePruningMethod subclass
        """
        return self._pruning_methods[idx]

    # Note: it's risky to try to change the container without explicitly
    # changing the masks computed on the module. We should not support
    # set and del item
    # def __setitem__(self, idx, method):
    #     # Can only reset the last item of the pruning history.
    #     # All other steps are immutable.
    #     # TODO: perhaps one day we might want to support intervening
    #     #    on the pruning history at any point, and recalculating all of
    #     #    the future pruning masks and amounts from this new insertion
    #     #    def reset_history_from(self, idx)

    #     # TODO: check that idx is an int (slices not supported)
    #     if not any(idx == _ for _ in [-1, len(self) - 1]):
    #         raise IndexError("Only the last item of the pruning history can "
    #             "be reset. All intermediate steps are immutable. Trying to "
    #             "reset item {} of PruningContainer of length {}". format(
    #                 idx, len(self)))

    #     # Without the logic above, this would erase all the history after
    #     # idx and restart writing history from there. Could this be an 
    #     # intentional scenario to support?
    #     self._pruning_methods = self._pruning_methods[:idx]
    #     self.add_pruning_method(method)

    # def __delitem__(self, idx):
    #     if isinstance(idx, slice):
    #         if idx.step is not None and idx.step != 1:
    #             raise IndexError("Deleting non-contiguous steps in the "
    #                 "pruning history is not supported")
    #         if not any(idx.stop == _ for _ in [-1, len(self) - 1]):
    #             raise IndexError("Deleting intermediate steps in the pruning "
    #                 "history is not supported. The slice must overlap with "
    #                 "the last element in the container.") 

    #         # assuming that the slice is equivalent to (idx.start, -1, 1),
    #         # delete everything after idx.start
    #         self._pruning_methods = self._pruning_methods[:idx.start]

    #     elif isinstance(idx, int):
    #         if idx >= len(self):
    #             raise IndexError("Index out of range")

    #         if not any(idx == _ for _ in [-1, len(self) - 1]):
    #             raise IndexError("Only the last item of the pruning history can"
    #             "be deleted. All intermediate steps are immutable. Trying to"
    #             "delete item {} of PruningContainer of length {}". format(
    #                 idx, len(self)))
            
    #         self._pruning_methods = self._pruning_methods[:idx]


    # @staticmethod
    # def apply(module, name, amount, *args):
    #     old_method = None
    #     found = 0
    #     method = PruningContainer(*args)

    #     # there should technically be only 1 hook with hook.name == name
    #     # assert this using `found`
    #     for k, hook in module._forward_pre_hooks.items():
    #         # if it exists, take existing thing, remove hook, then go thru normal thing
    #         if isinstance(hook, BasePruningMethod) and hook.name == name:
    #             old_method = hook
    #             module = remove_pruning(module, name)
    #             found += 1
    #     assert found <= 1 
        
    #     if old_method is not None:  # meaning that there was a hook
    #         # if the hook is already a pruning container, just add the new
    #         # pruning method to the container
    #         if isinstance(old_method, PruningContainer):
    #             old_method.add_pruning_method(method)
    #             method = old_method  # rename old_method --> method

    #         # if the hook is simply a single pruning method, create a 
    #         # container, add the old pruning method and the new one
    #         elif isinstance(old_method, BasePruningMethod):
    #             container = PruningContainer(name, old_method)
    #             container.add_pruning_method(method)
    #             method = container  # rename container --> method

    #     # original tensor, prior to this iteration of pruning
    #     orig = getattr(module, name)

    #     # temp remove tensor from parameter list
    #     del module._parameters[name]

    #     mask = method.compute_mask(orig)
    #     module.register_parameter(name + '_orig', torch.nn.Parameter(orig.data))

    #     # reparametrize
    #     module.register_buffer(name + '_mask', mask)
    #     setattr(module, name, method.apply_mask(module))

    #     # recompute function before every forward()
    #     module.register_forward_pre_hook(method)

    #     return method

    # def remove(self, module, name):
    #     r"""Removes the pruning reparameterization from a module. The pruned
    #     parameter named `name` remains permanently pruned, and the parameter
    #     named `name+'_orig'` is removed from the parameter list. Similarly,
    #     the buffer named `name+'_mask' is removed from the buffers.

    #     Note: 
    #         Pruning itself is NOT undone or reversed!
    #     """
    #     # unsure why I need to call this again here instead of just getting orig (backward?)
    #     weight = self.apply_mask(module)  # masked weights

    #     # delete and reset
    #     delattr(module, name)
    #     del module._parameters[name + '_orig']
    #     # TODO: do I delete the buffer too?
    #     del module._buffers[name + '_mask']
    #     module.register_parameter(name, torch.nn.Parameter(weight.data))



    def compute_mask(self, t):

        def _combine_masks(mask, method, t):
            """Compute new cumulative mask from old mask * new partial mask.
            The new partial mask should be computed on the entries that
            were not zeroed out by the old mask.

            Args:
                mask (torch.Tensor): mask from previous pruning iteration
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor being pruned (of same dimensions
                    as mask).
            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current 
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask
            partial_mask = method.compute_mask(t[mask == 1])
            new_mask[mask == 1] = partial_mask.float()
            return new_mask

        # init mask to all ones (nothing pruned)
        mask = torch.ones_like(t)

        # apply each method one at a time
        for method in self:
            mask = _combine_masks(mask, method, t)

        return mask

    def get_cumulative_amount(self):

        # init pruning amount to zero (nothing pruned)
        amount = 0

        # at each pruning iteration, remove `method.amount` from the
        # remaining `(1-amount)*100` % of tensor t
        for method in self:
            amount += (1 - amount) * method.amount

        return amount





class RandomPruningMethod(BasePruningMethod):

    def __init__(self, amount):
        """
        Args:
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the 
                absolute number of parameters to prune.
        """
        # super(RandomPruningMethod, self).__init__()

        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement() # TODO: "size" is misleading
        # Compute number of units to prune: amount if int, 
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = torch.ones_like(t)
        else:
            # TODO: torch.sparse?
            # Create random mask with nparams_nparams_toprune entries set to 0 
            prob = torch.rand_like(t)
            threshold = torch.kthvalue(prob.view(-1), k=nparams_toprune).values
            mask = prob > threshold
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the 
                absolute number of parameters to prune.
        """
        # this is here just for docstring generation for docs
        return super(RandomPruningMethod, cls).apply(module, name, amount=amount)

    # @staticmethod
    # def apply(module, name, amount):

        # # Check if a pruning method has already been applied to `module[name]`
        # old_method = None
        # found = 0
        # # there should technically be only 1 hook with hook.name == name
        # # assert this using `found`
        # for k, hook in module._forward_pre_hooks.items():
        #     # if it exists, take existing thing, remove hook, then go thru normal thing
        #     if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
        #         old_method = hook
        #         # reset the tensor reparametrization
        #         module = remove_pruning(module, name)
        #         found += 1
        # assert found <= 1, "Avoid adding multiple pruning hooks to the\
        #     same tensor {} of module {}. Use a PruningContainer.".format(
        #         name, module)

        # # Apply the new pruning method, either from scratch or on top of 
        # # the previous one.
        # method = RandomPruningMethod(amount)  # new pruning method
        
        # if old_method is not None:  # meaning that there was a hook
        #     # if the hook is already a pruning container, just add the new
        #     # pruning method to the container
        #     if isinstance(old_method, PruningContainer):
        #         old_method.add_pruning_method(method)
        #         method = old_method  # rename old_method --> method

        #     # if the hook is simply a single pruning method, create a 
        #     # container, add the old pruning method and the new one
        #     elif isinstance(old_method, BasePruningMethod):
        #         container = PruningContainer(old_method)
        #         container.add_pruning_method(method)
        #         method = container  # rename container --> method

        # # Have the pruning method remember what tensor it's been applied to
        # setattr(method, '_tensor_name', name)

        # # original tensor, prior to this iteration of pruning
        # orig = getattr(module, name)

        # # temp remove tensor from parameter list
        # del module._parameters[name]

        # mask = method.compute_mask(orig)
        # module.register_parameter(name + '_orig', torch.nn.Parameter(orig.data))

        # # reparametrize
        # module.register_buffer(name + '_mask', mask)
        # setattr(module, name, method.apply_mask(module))

        # # recompute function before every forward()
        # module.register_forward_pre_hook(method)

        # return method

    # def remove(self, module, name):
    #     r"""Removes the pruning reparameterization from a module. The pruned
    #     parameter named `name` remains permanently pruned, and the parameter
    #     named `name+'_orig'` is removed from the parameter list. Similarly,
    #     the buffer named `name+'_mask' is removed from the buffers.

    #     Note: 
    #         Pruning itself is NOT undone or reversed!
    #     """
    #     # unsure why I need to call this again here instead of just getting orig (backward?)
    #     weight = self.apply_mask(module)  # masked weights

    #     # delete and reset
    #     delattr(module, name)
    #     del module._parameters[name + '_orig']
    #     # TODO: do I delete the buffer too?
    #     del module._buffers[name + '_mask']
    #     module.register_parameter(name, torch.nn.Parameter(weight.data))


def random_pruning(module, name, amount):
    """Modifies module in place (and also return the modified module) 
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the 
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the 
    original (unpruned) parameter is stored in a new parameter named 
    `name+'_orig'`.
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the 
            absolute number of parameters to prune.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module, 
    """
    RandomPruningMethod.apply(module, name, amount)
    return module


def remove_pruning(module, name):
    r"""Removes the pruning reparameterization from a module. The pruned
    parameter named `name` remains permanently pruned, and the parameter
    named `name+'_orig'` is removed from the parameter list. Similarly,
    the buffer named `name+'_mask' is removed from the buffers.

    Note: 
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Example:
        >>> m = random_pruning(nn.Linear(5, 7), name='weight', amount=0.2)
        >>> remove_pruning(m, name='weight')
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("Pruning of '{}' not found in {}"
                     .format(name, module))

#TODO
def undo_pruning(module, name, steps=1):
    r"""

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        steps (int): number of pruning iterations to remove from the
            history (starting from the end). Must be <= than the number
            of pruning iterations that are present in the history.

    Example:
        >>> m = nn.Linear(5, 7)
        >>> random_pruning(m, name='weight', amount=0.2)
        >>> undo_pruning(m, name='weight', step=1)
    """
    raise NotImplementedError()

def _validate_pruning_amount_init(amount):
    """Validation helper to check the range of amount at init.
    
    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the 
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
                    integer. 
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
        Inspired by scikit-learn train_test_split.
    """
    amount_type = np.asarray(amount).dtype.kind

    if (amount_type == 'f' and (amount > 1. or amount < 0.)
        or amount_type == 'i' and amount < 0):
            raise ValueError("amount={} should either be a float in the "
                             "range [0, 1] or a non-negative integer"
                             "".format(amount))

    if amount_type not in ('i', 'f'):
        raise TypeError("Invalid type for amount: {}. Must be int or float."
                        "".format(amount))

def _validate_pruning_amount(amount, tensor_size):
    """Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (tensor_size).
    
    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the 
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
        Inspired by scikit-learn train_test_split.
    """
    amount_type = np.asarray(amount).dtype.kind

    if amount_type == 'i' and amount > tensor_size:
        raise ValueError("amount={} should be smaller than the number of "
                         "parameters to prune={}".format(amount, tensor_size))

def _compute_nparams_toprune(amount, tensor_size):
    """TODO
    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the 
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    Returns:
        int: the number of units to prune in the tensor
    """
    amount_type = np.asarray(amount).dtype.kind

    if amount_type == 'i':
        return amount
    elif amount_type == 'f':
        return round(amount * tensor_size)
    # incorrect type already checked in _validate_pruning_amount_init


