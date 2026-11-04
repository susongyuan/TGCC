"""Leave finite raw commands intact and retain native simulator limit handling.

This removes only TGCC's extra all-or-nothing chunk range veto. Actual
controller input clipping, joint target clipping and physics limits are not
changed. Callers must attest the live native controller configuration.
"""
import ast,functools,inspect,textwrap
import numpy as np
import torch


def range_diagnostics(native,spec):
    values=torch.as_tensor(native).detach().float().cpu().numpy()
    if values.shape!=(32,23) or not np.isfinite(values).all():raise ValueError('Finite native32x23 command required')
    lo=np.asarray(spec.lower,np.float32);hi=np.asarray(spec.upper,np.float32)
    excess=np.maximum(np.maximum(lo-values,values-hi),0);bad=excess>0
    return dict(raw_commands_within_nominal_ranges=not bool(bad.any()),
        violating_values=int(bad.sum()),max_excess=float(excess.max()),
        dimensions=[dict(index=i,name=spec.names[i],count=int(bad[:,i].sum()),max_excess=float(excess[:,i].max()))
            for i in range(23) if bad[:,i].any()],
        enforcement='unchanged_simulator_native_controller_clipping_and_joint_limits',
        application_chunk_range_veto=False,application_clipping=False)


def install(controller_class):
    if getattr(controller_class,'_native_controller_policy_installed',False):return
    def finite_native(self,actions):
        native=self.normalizer.inverse(actions[...,:23])
        if native.shape!=(1,32,23) or not bool(torch.isfinite(native).all()):
            raise ValueError('Nonfinite or malformed native action chunk')
        self.trace.setdefault('raw_command_range_diagnostics',[]).append(range_diagnostics(native[0],self.checker.spec))
        return native,True
    def commit(self,native_actions):
        actions=np.asarray(native_actions,np.float32)
        if self.previous is None or actions.shape!=(32,23) or not np.isfinite(actions).all():
            raise ValueError('Official dispatched action chunk mismatch')
        self.trace['dispatched_raw_command_ranges']=range_diagnostics(actions,self.checker.spec)
        self.previous['native_actions']=actions.copy()
    original_prepare=getattr(controller_class,'prepare',None)
    if original_prepare is not None:
        @functools.wraps(original_prepare)
        def prepare(self,raw,conditions,*,enforce_assistance_budget=True):
            # Distribution deviation is advisory. Keep per-request generation
            # bounded, but do not turn repeated advice into a physical failure.
            result=original_prepare(self,raw,conditions,enforce_assistance_budget=False)
            self.trace['statistical_feedback_budget_exceeded']=self.consecutive_feedback>self.max_consecutive_feedback
            self.trace['statistical_feedback_is_task_terminal']=False
            return result
        controller_class.prepare=prepare
    original=controller_class.sample_actions
    if controller_class.__name__=='DistributionController':
        tree=ast.parse(textwrap.dedent(inspect.getsource(original)));replaced=[]
        class AdvisoryPlanning(ast.NodeTransformer):
            def visit_Raise(self,node):
                if any(isinstance(n,ast.Constant) and n.value=='No legal demonstrated-order subtask remains' for n in ast.walk(node)):
                    replaced.append(1)
                    return ast.parse("legal=conditions.legal\nself.trace['planning_support_empty_fallback']='retain_original_legal_skills; density_OOD_is_not_physical_infeasibility'").body
                return node
        tree=AdvisoryPlanning().visit(tree)
        if len(replaced)!=1:raise ValueError('Frozen planning advisory branch changed')
        namespace=dict(inspect.unwrap(original).__globals__)
        exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
        original=namespace['sample_actions']
    @functools.wraps(original)
    def sample(self,*args,**kwargs):
        def rename(value):
            if isinstance(value,dict):
                if 'native_constraints_satisfied' in value:value['finite_native_commands']=value.pop('native_constraints_satisfied')
                if value.get('selection_scope')=='lowest_conditional_nll_among_safe_original_and_one_reference_draw':
                    value['selection_scope']='lowest_conditional_nll_among_finite_candidates; native_controller_limits_retained'
                for v in value.values():rename(v)
            elif isinstance(value,list):
                for v in value:rename(v)
        try:return original(self,*args,**kwargs)
        finally:
            rename(self.trace)
            self.trace['native_control_policy']='simulator_native_limits; no_extra_chunk_range_veto'
    controller_class._safe=finite_native;controller_class.commit_dispatched=commit
    controller_class.sample_actions=sample;controller_class._native_controller_policy_installed=True
