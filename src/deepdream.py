import tensorflow as tf

OCTAVES = 3
OCTAVE_SCALE = 1.4

def create_dream_model(model, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=model.input, outputs=outputs)

def calc_loss(dream_model, img):
    acts = dream_model(img)
    if not isinstance(acts, list):
        acts = [acts]
    return tf.reduce_sum([tf.reduce_mean(a) for a in acts])

def gradient_ascent(img, model, step_size, octaves=OCTAVES, octave_scale=OCTAVE_SCALE):
    base_shape = img.shape[1:3]
    oct_imgs = [img]
    for i in range(octaves - 1):
        new_size = [int(base_shape[0] / (octave_scale ** (i + 1))),
                    int(base_shape[1] / (octave_scale ** (i + 1)))]
        oct_imgs.append(tf.image.resize(img, new_size))

    detail = tf.zeros_like(oct_imgs[-1])
    for i in reversed(range(octaves)):
        oct_img = oct_imgs[i] + tf.image.resize(detail, oct_imgs[i].shape[1:3])
        oct_img = tf.Variable(oct_img)
        for _ in range(3):
            with tf.GradientTape() as tape:
                tape.watch(oct_img)
                loss = calc_loss(model, oct_img)
            grads = tape.gradient(loss, oct_img)
            grads /= tf.math.reduce_std(grads) + 1e-8
            oct_img.assign_add(step_size * grads)
        detail = oct_img - oct_imgs[i]
    return oct_img
