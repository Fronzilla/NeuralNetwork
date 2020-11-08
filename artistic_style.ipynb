{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.image import load_img, img_to_array\n",
    "import numpy as np\n",
    "from keras.applications import vgg19\n",
    "from keras import backend as K \n",
    "from scipy.optimizer import fmin_1_bfgs_b\n",
    "form scipy.misc import imsave\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_img_path = '/Users/alexeynikitin/Desktop/base_img.jpg'\n",
    "style_reference_image_path = '/Users/alexeynikitin/Desktop/test_image.jpg'\n",
    "\n",
    "width, height = load_img(target_img_path).size\n",
    "img_height = 400\n",
    "img_width = int(width * img_height / height)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_image(image_path):\n",
    "    img = load_img(image_path, target_size = (img_height, img_width))\n",
    "    img = img_to_array(img)\n",
    "    img = np.expand_dims(img, axis = 0)\n",
    "    img = vgg19.preprocess_input(img)\n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def deprocess_image(x):\n",
    "    x[:, :, 0] += 103.939\n",
    "    x[:, :, 1] += 116.779\n",
    "    x[:, :, 2] += 123.68\n",
    "    x = x[:, :, ::-1]\n",
    "    x = np.clip(x, 0, 255).astype('uint8')\n",
    "    return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_image = K.constant(preprocess_image(target_img_path))\n",
    "style_reference_image = K.constant(preprocess_image(\n",
    "    style_reference_image_path))\n",
    "\n",
    "combination_image = K.placeholder((1, img_height, img_width, 3))\n",
    "input_tensor = K.concatenate(\n",
    "    [target_image, style_reference_image, combination_image], axis = 0)\n",
    "\n",
    "model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet',\n",
    "                    include_top=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def content_loss(base, combination):\n",
    "    return K.sum(K.square(combination-base))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gram_matrix(x):\n",
    "    features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))\n",
    "    gram = K.dot(features, K.transpose(features))\n",
    "    return gram"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def style_loss(style, combination):\n",
    "    S = gram_matrix(style)\n",
    "    C = gram_matrix(combination)\n",
    "    channels = 3\n",
    "    size = img_height * img_width\n",
    "    return K.sum(K.square(S - C)) / (4. (channels ** 2) * (size ** 2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def total_variation_loss(x):\n",
    "    a = K.square(\n",
    "    x[:, :img_height - 1, :img_width - 1, :] -\n",
    "    x[:, 1:, :img_width -1, :])\n",
    "    b = K.square(\n",
    "    x[:, :img_height - 1, :img_width - 1, :] -\n",
    "    x[:, 1:, :img_height -1, 1:, :])\n",
    "    return K.sum(K.pow(a + b, 1.25))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'float' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-26-3f5111c30eab>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     21\u001b[0m     \u001b[0mstyle_reference_features\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlayer_features\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m     \u001b[0mcombination_features\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlayer_features\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 23\u001b[0;31m     \u001b[0msl\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mstyle_loss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstyle_reference_features\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mcombination_features\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     24\u001b[0m     \u001b[0mloss\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mstyle_weight\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstyle_layers\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0msl\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     25\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-14-d9ff894ca2dc>\u001b[0m in \u001b[0;36mstyle_loss\u001b[0;34m(style, combination)\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0mchannels\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0msize\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mimg_height\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mimg_width\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msquare\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mS\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mC\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m4.\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mchannels\u001b[0m \u001b[0;34m**\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0msize\u001b[0m \u001b[0;34m**\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m: 'float' object is not callable"
     ]
    }
   ],
   "source": [
    "outputs_dict = dict([layer.name, layer.output] for layer in model.layers)\n",
    "content_layer = 'block5_conv2'\n",
    "style_layers = ['block1_conv1',\n",
    "               'block2_conv1',\n",
    "               'block3_conv1',\n",
    "               'block4_conv1',\n",
    "               'block5_conv1']\n",
    "total_variation_weight = 1e-4\n",
    "style_weight = 1.\n",
    "content_weight = 0.025\n",
    "\n",
    "loss = K.variable(0.)\n",
    "layer_features = outputs_dict[content_layer]\n",
    "target_image_features = layer_features[0,:,:,:]\n",
    "combination_features = layer_features[2,:,:,:]\n",
    "loss += content_weight * content_loss(\n",
    "    target_image_features, combination_features)\n",
    "\n",
    "for layer_name in style_layers:\n",
    "    layer_features = outputs_dict[layer_name]\n",
    "    style_reference_features = layer_features[1,:,:,:]\n",
    "    combination_features = layer_features[2,:,:,:]\n",
    "    sl = style_loss(style_reference_features,combination_features)\n",
    "    loss += (style_weight / len(style_layers)) * sl\n",
    "\n",
    "loss += total_variation_weight * total_variation_loss(combination_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grads = K.gradient(loss, combination_image)[0]\n",
    "fetch_loss_and_grads = K.function([combination_image], [loss, grads])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Evaluator(object):\n",
    "    \n",
    "    def __init__(self):\n",
    "        self.loss_value = None\n",
    "        self.grads_values = None\n",
    "    \n",
    "    def loss(self, x):\n",
    "        assert self.loss_value is None\n",
    "        x = x.reshape((1, img_height, img_width,3))\n",
    "        outs = fetch_loss_and_grads([x])\n",
    "        loss_value outs[0]\n",
    "        grad_values = outs[1].flatten().astype('float64')\n",
    "        self.loss_value = loss_value\n",
    "        self.grad_values = grad_values\n",
    "        return self.loss_value\n",
    "    \n",
    "    def grads(self, x):\n",
    "        assert self.loss_value is None \n",
    "        grad_values = np.copy(self.grad_values)\n",
    "        self.loss_value = None\n",
    "        self.grads_values = None\n",
    "        return grad_values\n",
    "evaluator = Evaluator()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_prefix = 'my_result'\n",
    "iterations = 20\n",
    "\n",
    "x = preprocess_image(target_image_path)\n",
    "x = x.flatten()\n",
    "for i in range(iterations):\n",
    "    start_time = time.time()\n",
    "    x, min_val, info = fmin_1_bfgs_b(evaluator.loss,\n",
    "                                     x,fprime=evaluator.grads, maxfun=20)\n",
    "    \n",
    "img = x.copy().reshape((img_height, img_width, 3))\n",
    "img = deprocess_image(img)\n",
    "fname = result_prefix + '_at_iteration%d.png' % i\n",
    "imsave(fname, img)\n",
    "end_time = time.time()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
