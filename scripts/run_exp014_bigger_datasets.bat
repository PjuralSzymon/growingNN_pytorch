@echo off
rem Experiment 014: Monte Carlo 2 growth on CIFAR-100 and Tiny ImageNet.
rem Starts both runs in their own windows; each writes a log next to its results.
rem Stop Experiment 013 Phase 18 first: more than two runs at once can crash the 8 GB GPU.

cd /d "%~dp0.."

set DATA=experiments\data
set OUT=experiments\output\train_cifar10\runs\exp014_bigger_datasets\v1

if not exist "%DATA%\tiny-imagenet-200\train" (
    if exist "D:\repos\growingNN_pytorch\experiments\data\tiny-imagenet-200\train" (
        echo Copying Tiny ImageNet from the growingNN_pytorch checkout...
        xcopy /e /i /q "D:\repos\growingNN_pytorch\experiments\data\tiny-imagenet-200" "%DATA%\tiny-imagenet-200" >nul
    ) else (
        echo Tiny ImageNet not found locally; the script will download it.
    )
)

if not exist "%OUT%" mkdir "%OUT%"

start "Exp014 CIFAR-100" cmd /k python -u experiments\train_cifar10_exp014_bigger_datasets.py --dataset cifar100 ^> "%OUT%\cifar100.log" 2^>^&1
start "Exp014 Tiny ImageNet" cmd /k python -u experiments\train_cifar10_exp014_bigger_datasets.py --dataset tiny_imagenet ^> "%OUT%\tiny_imagenet.log" 2^>^&1

echo Started both runs. Logs:
echo   %OUT%\cifar100.log
echo   %OUT%\tiny_imagenet.log
