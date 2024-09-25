# Training Adapters

With Accelerate (Multi-GPU):

<code>accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.75 --dataset_fraction 0.1 --scheduler step --lr_step_size 40 --epochs 120 --lr 0.1
</code>

With Python (Single-GPU) :

<code>accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.75 --dataset_fraction 0.1 --scheduler step --lr_step_size 40 --epochs 120 --lr 0.1
</code>

### Parameters : 

<code>--arch</code> : Architecture of backbone model <br>
<code>--dataset</code> : Dataset to use <br>
<code>--outdir</code> : Directory to store trained adapter <br>
<code>--batch</code> : Batch Size to use for training/testing <br>
<code>--data_dir</code> : Dataset Directory <br>
<code>--noise_sd</code> : standard deviation of noise to use for training/testing <br>
<code>--dataset_fraction</code> : Fraction of dataset to use <br>
<code>--scheduler</code> : LR scheduler to use (cosine/step) <br>

# Certifying Adapters

<code>python certify_adapters.py --dataset cifar10 --base_classifier vit --sigma 1.0 --outfile certify_adapters_RS_1.0.txt --N 10000 --skip 20 --adapter /scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_1.0/1.0/ </code>


### Parameters : 


<code>--dataset</code> : Dataset to use <br>
<code>--base_classifier</code> : Architecture of backbone model <br>
<code>--sigma</code> : standard deviation of noise to use for certification <br>
<code>--outfile</code> : Output file to store certification result <br>
<code>--N</code> : N <br>
<code>--skip</code> : Number of images to skip before certifying the next image <br>
<code>--adapter</code> : Directory of the trained adapter <br>

# Training denoisers

<code>accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.1 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.1_0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 0.25 --workers 8 --epochs 150 --lr_step_size 50
</code>

# Certifying denoisers
python certify.py --dataset cifar10 --base_classifier vit --sigma 1.0 --outfile certify_denoiser_1.0.txt --denoiser /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_1.0/checkpoint.pth.tar --N 10000 --batch 512 --skip 20
