import kit.globals as G
from kit.path import join
from kit.jobs import plot_job_metrics

from CAPE.XVAE import get_model_dir_path, add_shortcut_to_artefacts


def run():
    if G.TASK.ID == "eval_plot_metrics":
        join(G.ENV.ARTEFACTS, "figures")
        model_id = G.TASK.MODEL_ID
        model_dir_path = get_model_dir_path(model_id)

        y_values = G.TASK.Y.split("+")
        for y_value in y_values:
            filename = join(model_dir_path, "figures", f"{model_id}_{y_value}.pdf")
            plot_job_metrics(
                filename,
                model_id,
                G.TASK.X,
                y_value,
                splits=True,
                xscale=G.TASK.X_SCALE,
                yscale=G.TASK.Y_SCALE,
            )

        add_shortcut_to_artefacts("figures", model_dir_path, model_id)
