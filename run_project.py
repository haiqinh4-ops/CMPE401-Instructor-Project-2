# CMPE 401 Project 2 — generated from notebook cells
# Run with Python after installing requirements; downloads datasets on first use.

import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

RUN_MODE = os.environ.get("CMPE401_MODE", "full")  # 'full' or 'smoke'
assert RUN_MODE in ("full", "smoke")
SEED = 42
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
print("TensorFlow:", tf.__version__, "Keras:", keras.__version__)
print("Mode:", RUN_MODE, "GPUs:", tf.config.list_physical_devices("GPU"))

def load_forda():
    base = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    paths = [keras.utils.get_file(fname=name, origin=base + name)
             for name in ("FordA_TRAIN.tsv", "FordA_TEST.tsv")]
    train = np.loadtxt(paths[0], delimiter="\t", dtype="float32")
    test = np.loadtxt(paths[1], delimiter="\t", dtype="float32")
    x_train, y_train = train[:, 1:, None], train[:, 0].astype("int32")
    x_test, y_test = test[:, 1:, None], test[:, 0].astype("int32")
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]
    if RUN_MODE == "smoke":
        x_train, y_train = x_train[:256], y_train[:256]
        x_test, y_test = x_test[:128], y_test[:128]
    print("FordA:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.25):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads,
                                  dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = layers.Add()([x, inputs])
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=int(inputs.shape[-1]), kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return layers.Add()([x, res])


def make_transformer(input_shape, num_classes):
    inp = keras.Input(shape=input_shape)
    x = inp
    for _ in range(4):  # official Keras baseline
        x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.25)
    # FordA has one channel; mean pooling across 500 steps erases the signal.
    # Flatten preserves all 500 time-step values for the classifier.
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="forda_transformer_baseline")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model


def plot_history(history, keys, title, path):
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        if key in history.history:
            ax.plot(history.history[key], label="training")
        if "val_" + key in history.history:
            ax.plot(history.history["val_" + key], label="validation")
        ax.set(xlabel="Epoch", ylabel=key, title=key.replace('_', ' ').title())
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

keras.backend.clear_session()
x_train, y_train, x_test, y_test = load_forda()
transformer = make_transformer(x_train.shape[1:], len(np.unique(y_train)))
transformer.summary()
start = time.perf_counter()
tr_history = transformer.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=150 if RUN_MODE == "full" else 2,
    batch_size=64,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                              restore_best_weights=True)],
    verbose=2,
)
tr_seconds = time.perf_counter() - start
tr_eval = transformer.evaluate(x_test, y_test, return_dict=True, verbose=0)
transformer_result = {
    "run_mode": RUN_MODE,
    "model": "Transformer classification baseline",
    "dataset": "FordA",
    "train_samples": int(len(x_train)),
    "test_samples": int(len(x_test)),
    "input_shape": list(x_train.shape[1:]),
    "parameters": int(transformer.count_params()),
    "epochs_run": len(tr_history.history["loss"]),
    "best_val_accuracy": float(max(tr_history.history["val_accuracy"])),
    "test_loss": float(tr_eval["loss"]),
    "test_accuracy": float(tr_eval["accuracy"]),
    "train_seconds": round(tr_seconds, 2),
}
(OUT_DIR / "transformer_baseline.json").write_text(
    json.dumps(transformer_result, indent=2), encoding="utf-8"
)
plot_history(tr_history, ["loss", "accuracy"], "FordA Transformer baseline",
             OUT_DIR / "transformer_learning_curves.png")
print(json.dumps(transformer_result, indent=2))

from zipfile import ZipFile

JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
FEATURES = [
    "p (mbar)", "T (degC)", "VPmax (mbar)", "VPdef (mbar)",
    "sh (g/kg)", "rho (g/m**3)", "wv (m/s)"
]
PAST = 720
FUTURE = 72
STEP = 6
SEQ_LEN = PAST // STEP


def load_jena_datasets():
    zippath = keras.utils.get_file(fname="jena_climate_2009_2016.csv.zip", origin=JENA_URL)
    with ZipFile(zippath) as zf:
        with zf.open("jena_climate_2009_2016.csv") as f:
            frame = pd.read_csv(f)
    raw = frame[FEATURES].to_numpy(dtype="float32")
    split = int(0.715 * len(raw))
    mean = raw[:split].mean(axis=0)
    std = raw[:split].std(axis=0)
    std = np.maximum(std, 1e-6)
    values = (raw - mean) / std
    train_x = values[:split]
    train_y = values[PAST + FUTURE:PAST + FUTURE + split, 1]
    val_x = values[split:len(values) - PAST - FUTURE]
    val_y = values[split + PAST + FUTURE:, 1]
    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)
    if RUN_MODE == "smoke":
        train_x, train_y = train_x[:4000], train_y[:4000]
        val_x, val_y = val_x[:1800], val_y[:1800]
    bs = 256 if RUN_MODE == "full" else 128
    ds_train = keras.utils.timeseries_dataset_from_array(
        train_x, train_y, sequence_length=SEQ_LEN, sampling_rate=STEP,
        batch_size=bs, shuffle=False)
    ds_val = keras.utils.timeseries_dataset_from_array(
        val_x, val_y, sequence_length=SEQ_LEN, sampling_rate=STEP,
        batch_size=bs, shuffle=False)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    batch_x, batch_y = next(iter(ds_train))
    print("Jena rows:", len(frame), "train/val raw rows:", len(train_x), len(val_x))
    print("Batch input:", batch_x.shape, "target:", batch_y.shape)
    print("Feature list:", FEATURES)
    return ds_train, ds_val, mean, std


def make_lstm(units=32, layers_count=1, learning_rate=1e-3):
    inp = keras.Input(shape=(SEQ_LEN, len(FEATURES)))
    if layers_count == 1:
        x = layers.LSTM(units)(inp)
    elif layers_count == 2:
        x = layers.LSTM(units, return_sequences=True)(inp)
        x = layers.LSTM(units)(x)
    else:
        raise ValueError("Supported layer counts: 1 or 2")
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

train_ds, val_ds, jena_mean, jena_std = load_jena_datasets()
experiment_configs = [
    {"experiment": "baseline", "units": 32, "layers_count": 1, "learning_rate": 0.001},
    {"experiment": "units_64", "units": 64, "layers_count": 1, "learning_rate": 0.001},
    {"experiment": "stacked_2_layers", "units": 32, "layers_count": 2, "learning_rate": 0.001},
    {"experiment": "learning_rate_0.0005", "units": 32, "layers_count": 1, "learning_rate": 0.0005},
]
records = []
learning_curves = {}

for config in experiment_configs:
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = make_lstm(config["units"], config["layers_count"], config["learning_rate"])
    print("\n==", config["experiment"], "| parameters:", model.count_params(), "==")
    begin = time.perf_counter()
    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=10 if RUN_MODE == "full" else 2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                 restore_best_weights=True)],
        verbose=2,
    )
    seconds = time.perf_counter() - begin
    evaluation = model.evaluate(val_ds, return_dict=True, verbose=0)
    record = {
        **config,
        "mode": RUN_MODE,
        "parameters": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_val_mse": float(min(hist.history["val_loss"])),
        "evaluation_val_mse": float(evaluation["loss"]),
        "evaluation_val_mae": float(evaluation["mae"]),
        "train_seconds": round(seconds, 2),
    }
    records.append(record)
    learning_curves[config["experiment"]] = hist.history

results = pd.DataFrame(records)
results.to_csv(OUT_DIR / "lstm_benchmark.csv", index=False)
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in learning_curves.items():
    axes[0].plot(h["val_loss"], marker="o", label=name)
axes[0].set(title="Jena LSTM validation MSE by epoch", xlabel="Epoch", ylabel="MSE")
axes[0].grid(alpha=0.2)
axes[0].legend(fontsize=8)
axes[1].bar(results["experiment"], results["evaluation_val_mse"])
axes[1].set(title="LSTM benchmark (lower is better)", ylabel="Validation MSE")
axes[1].tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUT_DIR / "lstm_benchmark.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

baseline = results.loc[results["experiment"] == "baseline"].iloc[0]
modifications = results.loc[results["experiment"] != "baseline"].copy()
best_modified = modifications.sort_values("evaluation_val_mse").iloc[0]
relative_pct = 100 * (baseline["evaluation_val_mse"] - best_modified["evaluation_val_mse"]) / baseline["evaluation_val_mse"]
mod_label = {
    "units_64": "increasing the LSTM hidden size from 32 to 64",
    "stacked_2_layers": "stacking a second 32-unit LSTM layer",
    "learning_rate_0.0005": "reducing the learning rate from 0.001 to 0.0005",
}
change = mod_label[best_modified["experiment"]]
effect = "improved" if relative_pct > 0 else "did not improve"

lines = [
    "# CMPE 401 Project 2 — Experiment Summary",
    "",
    f"**Execution mode:** `{RUN_MODE}`",
    "" if RUN_MODE == "full" else "**WARNING: SMOKE TEST ONLY — NOT SUBMISSION RESULTS**",
    "",
    "## Task 1 — Baseline reproduction",
    f"- **Transformer / FordA:** input {transformer_result['input_shape']}; "
    f"{transformer_result['parameters']:,} parameters; "
    f"{transformer_result['epochs_run']} epochs; "
    f"test accuracy **{transformer_result['test_accuracy']:.4f}**; "
    f"test cross-entropy loss **{transformer_result['test_loss']:.4f}**.",
    f"- **LSTM / Jena Climate:** input (120, 7); baseline LSTM(32); "
    f"{int(baseline['parameters']):,} parameters; "
    f"validation MSE **{baseline['evaluation_val_mse']:.4f}**; "
    f"validation MAE **{baseline['evaluation_val_mae']:.4f}**.",
    "",
    "## Task 2 and 3 — Controlled modifications / benchmark",
    "",
    "| Experiment | Hidden units | LSTM layers | Learning rate | Val MSE | Val MAE | Params | Training (s) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for _, r in results.iterrows():
    lines.append(f"| {r['experiment']} | {int(r['units'])} | {int(r['layers_count'])} | "
                 f"{r['learning_rate']:.4f} | {r['evaluation_val_mse']:.4f} | "
                 f"{r['evaluation_val_mae']:.4f} | {int(r['parameters']):,} | "
                 f"{r['train_seconds']:.1f} |")
lines.extend([
    "",
    "The three experiments change one hyperparameter/architectural choice at a time. "
    "The Transformer and LSTM scores are **not directly comparable**, because they use different datasets and tasks.",
    "",
    "## Task 4 — Questions (draft answers; review in your own words)",
    "",
    "### Q1. Which model did you find easier to understand and why?",
    "**Suggested personal response:** I found the LSTM model easier to understand because the "
    "Keras baseline uses a single LSTM layer followed by one Dense output layer. It also directly "
    "connects a sequence of weather measurements to a single future temperature, whereas "
    "the Transformer contains multiple attention and residual blocks. [Edit this to match your experience.]",
    "",
    "### Q2. What improvement did you try, and what did you learn from it?",
    f"I tested three independent changes: 64 hidden units, two stacked LSTM layers, and "
    f"a learning rate of 0.0005. The modified run with the lowest validation MSE was "
    f"**{best_modified['experiment']}** ({change}). It {effect} validation MSE relative to the "
    f"baseline by **{abs(relative_pct):.2f}%** (baseline {baseline['evaluation_val_mse']:.4f}; "
    f"modified {best_modified['evaluation_val_mse']:.4f}). This shows that model settings "
    f"should be evaluated empirically rather than assuming more complexity is always beneficial.",
    "",
    "## Figures", "",
    "![Transformer learning curves](transformer_learning_curves.png)",
    "![LSTM benchmark](lstm_benchmark.png)",
    "",
    "## Limitations",
    "- The official Jena tutorial provides training and validation splits but no independent test split. "
    "Reported forecasting scores are validation metrics.",
    "- Results may differ between runs or hardware/software versions despite setting a seed.",
    "- These are single-seed experiments; repeated-seed evaluation would strengthen conclusions.",
    "",
    "## Sources",
    "- Transformer: https://keras.io/examples/timeseries/timeseries_classification_transformer/",
    "- LSTM: https://keras.io/examples/timeseries/timeseries_weather_forecasting/",
])
summary_text = "\n".join(lines) + "\n"
(OUT_DIR / "PROJECT_SUMMARY.md").write_text(summary_text, encoding="utf-8")
print("Saved summary and figures to", OUT_DIR.resolve())
print("\n".join(lines[:18]))

import shutil
archive = shutil.make_archive("CMPE401_Project2_outputs", "zip", root_dir=str(OUT_DIR))
print("Created", archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print("Not in Colab; ZIP is available at", archive)
