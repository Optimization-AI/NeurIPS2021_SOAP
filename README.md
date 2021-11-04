# Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/pdf/2104.08736.pdf)

This is the official implementation of the paper "**Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence**" published on **Neurips2021**. 


Benchmark Datasets
---------
**Image**: CIFAR10, CIFAR100, Melanoma \
**Graph**: HIV, MUV, AICures

Package
----------
The main algorithm **SOAP** has been implemented in [LibAUC](https://github.com/Optimization-AI/LibAUC/), with 
```python
>>> from libauc.optimizers import SOAP_SGD, SOAP_ADAM
```
You can design your own loss. The following is a usecase:
```python
pip install libauc
>>> #import library
>>> from libauc.losses import APLoss_SH
>>> from libauc.optimizers import SOAP_SGD, SOAP_ADAM
...
>>> #define loss
>>> Loss = APLoss_SH()
>>> optimizer = SOAP_ADAM()
...
>>> #training
>>> model.train()
>>> for index, data, targets in trainloader:
        data, targets  = data.cuda(), targets.cuda()
        logits = model(data)
	      preds = torch.sigmoid(logits)
        loss = Loss(preds, targets, index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```




Citation
---------
If you find this repo helpful, please cite the following paper:
```
@article{qi2021stochastic,
  title={Stochastic Optimization of Area Under Precision-Recall Curve for Deep Learning with Provable Convergence},
  author={Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao},
  journal={arXiv preprint arXiv:2104.08736},
  year={2021}
}
```

Contact
----------
If you have any questions, please contact us @ [Qi Qi](https://qiqi-helloworld.github.io/) [qi-qi@uiowa.edu] , and [Tianbao Yang](https://homepage.cs.uiowa.edu/~tyng/) [tianbao-yang@uiowa.edu] or please open a new issue in the Github. 
