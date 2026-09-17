"""system.doctor: dependency check + platform-correct fix planning.

Platform-specific torch/rfdetr planning lives in horos.api.install and is
covered by test_install_plan.py; here we test doctor's own report and the
delegation. `_plan_fixes` probes the live environment for GPU/torch-build
state, so assertions are membership-based (extra mismatch-repair commands may
legitimately appear on some machines).
"""

from horos.api.install import RFDETR_SPEC
from horos.api.system import _RUNTIME_DEPS, _plan_fixes, doctor_report
from horos.core.platform_info import PlatformInfo


def _plat(os_family="linux", arch="x86_64", is_jetson=False):
    return PlatformInfo(
        os_family=os_family, arch=arch, is_jetson=is_jetson, python_version="3.10.6"
    )


def test_report_on_current_env():
    report = doctor_report()
    names = [d.name for d in report.dependencies]
    assert {"pydantic", "flask", "torch", "rfdetr", "transformers"} <= set(names)
    # the lightweight core (`pip install horos`) is always complete in dev/CI
    light = {"pydantic", "flask", "yaml", "PIL"}
    assert all(d.ok for d in report.dependencies if d.name in light)
    # a broken or incomplete ML stack must always come with a plan
    if not report.ok:
        assert report.fix_commands or report.manual_actions
    else:
        assert report.fix_commands == [] and report.manual_actions == []


def test_missing_rfdetr_plans_pinned_install_with_training_stack():
    commands, manual = _plan_fixes(["rfdetr"], _plat())
    assert [RFDETR_SPEC] in commands and manual == []


def test_missing_training_stack_alone_reinstalls_the_extra():
    # rfdetr installed without [train] (e.g. an old horos env): fix via the extra
    commands, manual = _plan_fixes(["pytorch_lightning"], _plat())
    assert [RFDETR_SPEC] in commands and manual == []


def test_missing_albumentations_plans_the_pin():
    # `horos install` once skipped it, so training failed at the first step
    # on a "healthy" doctor report — doctor must report and plan it too
    assert "albumentations" in [name for name, _ in _RUNTIME_DEPS]
    commands, manual = _plan_fixes(["albumentations"], _plat())
    assert ["albumentations==2.0.8"] in commands and manual == []


def test_missing_transformers_plans_the_owlv2_range():
    commands, _ = _plan_fixes(["transformers"], _plat())
    assert ["transformers>=5.1.0,<6"] in commands


def test_jetson_never_automates_torch():
    commands, manual = _plan_fixes(["torch", "rfdetr"], _plat(arch="aarch64", is_jetson=True))
    flat = [arg for command in commands for arg in command]
    assert "torch" not in flat  # never pip-install torch on Jetson (§4)
    assert ["rfdetr==1.9.4", "--no-deps"] in commands  # cannot drag torch in
    assert any("JetPack" in m for m in manual)


def test_light_deps_use_their_spec():
    commands, _ = _plan_fixes(["flask"], _plat())
    assert ["flask>=3.0,<4"] in commands


# --------------------------------------------------------------- ROCm (AMD)
def _fake_cpu_torch_on(monkeypatch, *, nvidia=None, amd=None, arch=None):
    """Pretend the live machine has a complete stack built for no accelerator.

    doctor_report probes the real environment, so the GPU-mismatch arms can
    only be reached by standing in for those probes.
    """
    import horos.backends.env as env_mod
    from horos.api import install as install_mod
    from horos.api import system as sys_mod

    # The platform has to be stood in for as well, not just the probes:
    # doctor_report() calls detect_platform() itself, and plan_install()
    # drops the AMD GPU outright on macOS and Jetson (correctly — ROCm has
    # no build there). Without this the ROCm arms are unreachable on a Mac
    # and these tests only pass on the Linux/Windows CI runners.
    plat = _plat(os_family="windows", arch="AMD64")
    monkeypatch.setattr(sys_mod, "detect_platform", lambda: plat)
    monkeypatch.setattr(sys_mod, "_installed_version", lambda name: "1.0")
    monkeypatch.setattr(sys_mod, "torch_is_cpu_build", lambda: True)
    monkeypatch.setattr(sys_mod, "detect_cuda_version", lambda: nvidia)
    monkeypatch.setattr(sys_mod, "detect_amd_gpu", lambda: amd)
    monkeypatch.setattr(sys_mod, "detect_rocm_arch", lambda: arch)
    # the planner behind fix_commands probes independently
    monkeypatch.setattr(install_mod, "detect_amd_gpu", lambda: amd)
    monkeypatch.setattr(install_mod, "detect_rocm_arch", lambda: arch)
    monkeypatch.setattr(install_mod, "detect_cuda_version", lambda: nvidia)
    monkeypatch.setattr(install_mod, "torch_is_cpu_build", lambda: True)
    monkeypatch.setattr(
        env_mod,
        "check_environment",
        lambda emit_warnings=True: env_mod.EnvReport(
            platform=plat,
            torch_version="2.14.0+cpu",
            cuda_available=False,
            mps_available=False,
            warnings=[],
        ),
    )


def test_an_idle_amd_gpu_is_reported_instead_of_environment_ok(monkeypatch):
    # PyPI has no AMD torch, so a CPU build is what an AMD box gets by
    # default. Saying "Environment OK" here is the silent CPU fallback §4
    # forbids: the user trains on CPU and never finds out why it is slow.
    _fake_cpu_torch_on(monkeypatch, amd="AMD Radeon RX 9070 XT", arch="gfx1201")
    report = doctor_report()
    assert not report.ok
    torch_dep = next(d for d in report.dependencies if d.name == "torch")
    assert not torch_dep.ok
    assert "RX 9070 XT" in torch_dep.note and "gfx1201" in torch_dep.note
    # the architecture is known, so this is repairable: `doctor --fix` does it
    assert any("device-gfx1201" in arg
               for command in report.fix_commands for arg in command)
    assert report.manual_actions == []


def test_an_undetectable_architecture_becomes_a_manual_step(monkeypatch):
    # nothing safe to install, so doctor asks rather than guessing a wheel
    _fake_cpu_torch_on(monkeypatch, amd="AMD Radeon RX 9070 XT", arch=None)
    report = doctor_report()
    assert not report.ok
    torch_dep = next(d for d in report.dependencies if d.name == "torch")
    assert "architecture unknown" in torch_dep.note
    action = next(a for a in report.manual_actions if "RX 9070 XT" in a)
    assert "HOROS_ROCM_ARCH" in action
    assert not any("amd.com" in arg
                   for command in report.fix_commands for arg in command)


def test_a_cpu_box_with_no_gpu_at_all_stays_ok(monkeypatch):
    _fake_cpu_torch_on(monkeypatch, nvidia=None, amd=None)
    report = doctor_report()
    assert report.ok
    assert report.manual_actions == []


def test_an_nvidia_gpu_still_takes_priority_over_the_amd_arm(monkeypatch):
    # a machine with both: the CUDA path is the one horos knows how to plan
    _fake_cpu_torch_on(
        monkeypatch, nvidia=(13, 0), amd="AMD Radeon RX 9070 XT", arch="gfx1201"
    )
    report = doctor_report()
    torch_dep = next(d for d in report.dependencies if d.name == "torch")
    assert "NVIDIA" in torch_dep.note
    assert not any("amd.com" in arg
                   for command in report.fix_commands for arg in command)
