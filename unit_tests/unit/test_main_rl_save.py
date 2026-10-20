import os
import errno
import stat
import pytest
from unittest.mock import MagicMock, patch
import misc.setup_rl as setup_rl

def _make_training_engine(call_log, label):
    remote = MagicMock(side_effect=lambda **kwargs: call_log.append(label) or f"{label}_ref")
    state_remote = MagicMock(side_effect=lambda *a, **kw: f"{label}_state_ref")
    return SimpleNamespace(
        save_checkpoint=SimpleNamespace(remote=remote),
        save_engine_state=SimpleNamespace(remote=state_remote),
    )


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_save_checkpoint_waits_for_engine_saves_before_tokenizer_write(tmp_path):
    call_log = []
    logger = MagicMock()
    tokenizer = MagicMock()
    tokenizer.save_pretrained.side_effect = lambda path: call_log.append("tokenizer")
    engines = [
        _make_training_engine(call_log, "engine0"),
        _make_training_engine(call_log, "engine1"),
    ]

    def _wait_for_saves(refs, timeout, description, logger):
        call_log.append("wait")
        return [True for _ in refs]

    with patch.object(setup_rl, "ray_get_with_timeout", side_effect=_wait_for_saves), \
         patch.object(setup_rl.ray, "nodes", return_value=[{"Alive": True, "Resources": {"GPU": 2}}]):
        model_path = setup_rl.save_checkpoint(
            epoch=0,
            version=7,
            global_step=42,
            tokenizer=tokenizer,
            training_engines=engines,
            checkpoint_dir=str(tmp_path),
            experiment_id="exp4",
            rank=0,
            logger=logger,
            save_timeout=30,
        )

    assert call_log == ["engine0", "engine1", "wait", "wait", "tokenizer"]
    assert model_path == os.path.join(str(tmp_path), "exp4", "iter000001_v000007")
    tokenizer.save_pretrained.assert_called_once_with(model_path)
    # only the shared-filesystem warning is under test; a mount that rejects dir fsync may add its own
    assert not any("shared filesystem" in c.args[0] for c in logger.warning.call_args_list)


def test_save_checkpoint_warns_on_multi_gpu_node_cluster(tmp_path):
    logger = MagicMock()
    tokenizer = MagicMock()
    engines = [_make_training_engine([], "engine0")]
    nodes = [
        {"Alive": True, "Resources": {"CPU": 8, "GPU": 4}},
        {"Alive": True, "Resources": {"CPU": 8, "GPU": 4}},
    ]

    with patch.object(setup_rl, "ray_get_with_timeout", return_value=[True]), \
         patch.object(setup_rl.ray, "nodes", return_value=nodes):
        setup_rl.save_checkpoint(
            epoch=0,
            version=1,
            global_step=10,
            tokenizer=tokenizer,
            training_engines=engines,
            checkpoint_dir=str(tmp_path),
            experiment_id="exp4",
            rank=0,
            logger=logger,
            save_timeout=30,
        )

    assert sum("shared filesystem" in c.args[0] for c in logger.warning.call_args_list) == 1


def test_save_checkpoint_removes_stale_marker_before_engines_write(tmp_path):
    '''Re-saving into a tag directory a previous run completed must drop the old CHECKPOINT_COMPLETE
       and training_state.json before any shard is written, so a crash mid-save never leaves a
       half-overwritten checkpoint marked complete.'''
    model_path = os.path.join(str(tmp_path), "exp4", "iter000001_v000007")
    os.makedirs(model_path)
    for stale in ("CHECKPOINT_COMPLETE", "training_state.json"):
        with open(os.path.join(model_path, stale), "w") as f:
            f.write("old run")

    seen_during_save = []
    def remote(**kwargs):
        seen_during_save.append(sorted(os.listdir(kwargs["output_dir"])))
        return "engine0_ref"

    engine = SimpleNamespace(save_checkpoint=SimpleNamespace(remote=remote),
                             save_engine_state=SimpleNamespace(remote=MagicMock(return_value="engine0_state_ref")))

    with patch.object(setup_rl, "ray_get_with_timeout", return_value=[True]), \
         patch.object(setup_rl.ray, "nodes", return_value=[{"Alive": True, "Resources": {"GPU": 1}}]):
        setup_rl.save_checkpoint(
            epoch=0,
            version=7,
            global_step=42,
            tokenizer=MagicMock(),
            training_engines=[engine],
            checkpoint_dir=str(tmp_path),
            experiment_id="exp4",
            rank=0,
            logger=MagicMock(),
            save_timeout=30,
        )

    # neither stale file was visible while the engine was writing its shards
    assert seen_during_save == [[]]
    # this run wrote fresh copies at the end
    assert os.path.exists(os.path.join(model_path, "CHECKPOINT_COMPLETE"))
    with open(os.path.join(model_path, "training_state.json")) as f:
        assert '"global_step": 42' in f.read()


def _save_once(tmp_path, engine, logger):
    with patch.object(setup_rl, "ray_get_with_timeout", return_value=[True]), \
         patch.object(setup_rl.ray, "nodes", return_value=[{"Alive": True, "Resources": {"GPU": 1}}]):
        return setup_rl.save_checkpoint(epoch=0, version=7, global_step=42, tokenizer=MagicMock(),
                                        training_engines=[engine], checkpoint_dir=str(tmp_path),
                                        experiment_id="exp4", rank=0, logger=logger, save_timeout=30)


def test_save_checkpoint_fsyncs_the_dir_before_any_shard_is_written(tmp_path, monkeypatch):
    '''The unlinks of the stale files are made durable with a directory fsync before the engines write.'''
    model_path = os.path.join(str(tmp_path), "exp4", "iter000001_v000007")
    os.makedirs(model_path)
    open(os.path.join(model_path, "CHECKPOINT_COMPLETE"), "w").close()

    events, real_fsync = [], os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append(("dir_fsync", sorted(os.listdir(model_path))))
        return real_fsync(fd)

    monkeypatch.setattr(setup_rl.os, "fsync", fsync)
    def remote(**kwargs):
        events.append(("engine_save", None))
        return "engine0_ref"

    engine = SimpleNamespace(save_checkpoint=SimpleNamespace(remote=remote),
                             save_engine_state=SimpleNamespace(remote=MagicMock(return_value="engine0_state_ref")))
    _save_once(tmp_path, engine, MagicMock())
    assert events[:2] == [("dir_fsync", []), ("engine_save", None)]


def test_save_checkpoint_survives_a_mount_that_rejects_dir_fsync(tmp_path, monkeypatch):
    '''Some mounts reject fsync on a directory fd. The unlinks already happened, so the checkpoint must
       still complete, with a warning.'''
    real_fsync = os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(setup_rl.os, "fsync", fsync)
    logger = MagicMock()
    engine = SimpleNamespace(save_checkpoint=SimpleNamespace(remote=MagicMock(return_value="engine0_ref")),
                             save_engine_state=SimpleNamespace(remote=MagicMock(return_value="engine0_state_ref")))
    model_path = _save_once(tmp_path, engine, logger)
    assert os.path.exists(os.path.join(model_path, "CHECKPOINT_COMPLETE"))
    logger.warning.assert_called_once()
    assert "Could not fsync checkpoint dir" in logger.warning.call_args[0][0]


def test_save_checkpoint_stops_before_any_shard_when_a_stale_file_cannot_be_removed(tmp_path, monkeypatch):
    '''Only "file not found" is ignored when removing the stale files. Any other error (e.g. a permission
       error) stops the save before an engine writes, so the old checkpoint is left as it was.'''
    model_path = os.path.join(str(tmp_path), "exp4", "iter000001_v000007")
    os.makedirs(model_path)
    open(os.path.join(model_path, "CHECKPOINT_COMPLETE"), "w").close()

    def remove(path):
        raise PermissionError(errno.EACCES, "Permission denied", path)

    monkeypatch.setattr(setup_rl.os, "remove", remove)
    remote = MagicMock(return_value="engine0_ref")
    engine = SimpleNamespace(save_checkpoint=SimpleNamespace(remote=remote),
                             save_engine_state=SimpleNamespace(remote=MagicMock(return_value="engine0_state_ref")))
    with pytest.raises(PermissionError):
        _save_once(tmp_path, engine, MagicMock())

    remote.assert_not_called()
    assert os.path.exists(os.path.join(model_path, "CHECKPOINT_COMPLETE"))
