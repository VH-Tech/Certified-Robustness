from analyze import plot_certified_accuracy_per_sigma_best_model, Line, ApproximateAccuracy
import sys

certification_result_with_denoiser = sys.argv[1]
certification_result_without_denoiser = sys.argv[2]
star = sys.argv[3]

sigma = float(certification_result_with_denoiser.split('_')[3]) #sigma_0.75_no_denoiser

plot_certified_accuracy_per_sigma_best_model(
    f"{certification_result_with_denoiser.split('/')[0]}/certification_output/{sigma}_plot_{star}", 'With vs Without Denoiser', 1.0,
    methods=
        [Line(ApproximateAccuracy(certification_result_with_denoiser), "$\sigma = 0.12$")],
    label='With Denoiser',
    methods_base=
        [Line(ApproximateAccuracy(certification_result_without_denoiser), "$\sigma = 0.12$")], 
    label_base='Without Denoiser',
    sigmas=[sigma])
