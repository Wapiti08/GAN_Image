# GAN_Image
The research on black_box attacks to images based on GAN model

## Tutorial:
[TF2](https://www.kaggle.com/vikramtiwari/tf2-tutorials-keras-save-and-restore-models/data)

[obfuscated-gradients](https://github.com/anishathalye/obfuscated-gradients)

## Reference

[Paper Summary: Practical Black-Box Attacks against Machine Learning](https://medium.com/@hyponymous/paper-summary-practical-black-box-attacks-against-machine-learning-9f0b3a58e956)

[Adversarial Attacks and Defences for Convolutional Neural Networks](https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7)

[Getting to know a black-box model](https://towardsdatascience.com/getting-to-know-a-black-box-model-374e180589ce)

[How Adversarial Attacks Work](https://blog.ycombinator.com/how-adversarial-attacks-work/) --- Important

[Know Your Adversary: Understanding Adversarial Examples (Part 1/2)](https://towardsdatascience.com/know-your-adversary-understanding-adversarial-examples-part-1-2-63af4c2f5830)

## Upload
Please fork this repository, modify as you like, then send me a pull request. I will check to integrate.

## 说明

- local_search_attack:

参考

- single_pixle_av:

参考

- single_attack_attack:
核心代码

- solution_template （来自于之前我参加的一个比赛）:
定向攻击的例子 --- 未完成
目前有生成的模型

- one-pixel-attack:

error:
```
Could not import PIL.Image. The use of `load_img` requires PIL
```

    - What to do next:

        - Change the targeted.py:

        1. read image from outside images/test folder with for loop

        2. Change the way to targeted label to image
