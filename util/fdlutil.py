import importlib.util
import os
import sys
from pylab import *
import matplotlib as mpl

# Use tkAgg when plotting to a window, Agg when to a file
# #### mpl.use('TkAgg')  # Don't use this unless emergency. More trouble than it's worth
mpl.use('Agg')


def quick_imshow(nrows, ncols=1, images=None, titles=None, colorbar=True, colormap='jet',
                 vmax=None, vmin=None, figsize=None, figtitle=None, visibleaxis=True,
                 saveas='/home/ubuntu/tempimshow.png', tight=False, dpi=250.0):
    """-------------------------------------------------------------------------
    Desc.:      convenience function that make subplots of imshow
    Args.:      nrows - number of rows
                ncols - number of cols
                images - list of images
                titles - list of titles
                vmax - tuple of vmax for the colormap. If scalar,
                        the same value is used for all subplots. If one
                        of the entries is None, no colormap for that
                        subplot will be drawn.
                 vmin - tuple of vmin
    Returns:    f - the figure handle
                axes - axes or array of axes objects
                caxes - tuple of axes image
    -------------------------------------------------------------------------"""
    if isinstance(nrows, np.ndarray):
        images = nrows
        nrows = 1
        ncols = 1

    if figsize == None:
        # 1.0 translates to 100 pixels of the figure
        s = 5.0
        if figtitle:
            figsize = (s * ncols, s * nrows + 0.5)
        else:
            figsize = (s * ncols, s * nrows)

    if nrows == ncols == 1:
        if isinstance(images, list):
            images = images[0]
        f, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(images, cmap=colormap, vmax=vmax, vmin=vmin)
        if colorbar:
            f.colorbar(cax, ax=ax)
        if titles != None:
            ax.set_title(titles)
        if figtitle != None:
            f.suptitle(figtitle)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        if tight:
            plt.tight_layout()
        if len(saveas) > 0:
            dirname = os.path.dirname(saveas)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(saveas)
        return f, ax, cax

    f, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    caxes = []
    i = 0
    for ax, img in zip(axes.flat, images):
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=vmin[i])
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=0)
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif vmax is None and vmin is None:
            cax = ax.imshow(img, cmap=colormap)
        else:
            cax = ax.imshow(img, cmap=colormap, vmax=vmax, vmin=vmin)
        if titles != None:
            ax.set_title(titles[i])
        if colorbar:
            f.colorbar(cax, ax=ax)
        caxes.append(cax)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        i = i + 1
    if figtitle != None:
        f.suptitle(figtitle)
    if tight:
        plt.tight_layout()
    if len(saveas) > 0:
        dirname = os.path.dirname(saveas)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(saveas)
    return f, axes, tuple(caxes)


def update_subplots(images, caxes, f=None, axes=None, indices=(), vmax=None,
                    vmin=None):
    """-------------------------------------------------------------------------
    Desc.:      update subplots in a figure
    Args.:      images  - new images to plot
                caxes   - caxes returned at figure creation
                indices - specific indices of subplots to be updated
    Returns:
    -------------------------------------------------------------------------"""
    for i in range(len(images)):
        if len(indices) > 0:
            ind = indices[i]
        else:
            ind = i
        img = images[i]
        caxes[ind].set_data(img)
        cbar = caxes[ind].colorbar
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cbar.set_clim([vmin[i], vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cbar.set_clim([0, vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif vmax is None and vmin is None:
            cbar.set_clim([img.min(), img.max()])
        else:
            cbar.set_clim([vmin, vmax])
        cbar.update_normal(caxes[ind])
    pause(0.01)
    tight_layout()


def slide_show(image, dt=0.01, vmax=None, vmin=None):
    """
    Slide show for visualizing an image volume. Image is (w, h, d)
    :param image: (w, h, d), slides are 2D images along the depth axis
    :param dt:
    :param vmax:
    :param vmin:
    :return:
    """
    if image.dtype == bool:
        image *= 1.0
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()
    plt.ion()
    plt.figure()
    for i in range(image.shape[2]):
        plt.cla()
        cax = plt.imshow(image[:, :, i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(str('Slice: %i/%i' % (i, image.shape[2] - 1)))
        if i == 0:
            cf = plt.gcf()
            ca = plt.gca()
            cf.colorbar(cax, ax=ca)
        plt.pause(dt)
        plt.draw()


def quick_collage(images, nrows=3, ncols=2, normalize=False, figsize=(20.0, 10.0), figtitle=None, colorbar=True,
                  tight=True, saveas='/home/ubuntu/tempcollage.png'):
    def zero_to_one(x):
        if x.min() == x.max():
            return x - x.min()
        return (x.astype(float) - x.min()) / (x.max() - x.min())
    # Normalize every image
    if isinstance(images, np.ndarray):
        images = [images]
    # Check the shape and make sure everything is float
    img_shp = images[0].shape
    if normalize:
        images = [zero_to_one(image) for image in images]
        vmax, vmin = 1.0, 0.0
    else:
        vmax, vmin = max([img.max() for img in images]), min(
            [img.min() for img in images])
    # Highlight the boundaries
    for i in range(0, len(images) - 1):
        images[i] = np.hstack(
            [images[i], np.full((img_shp[0], 1, img_shp[2]), np.nan)])
    collage = np.hstack(images)
    # Determine slice depth
    depth = collage.shape[2]
    n_slices = nrows * ncols
    z = [int(depth / (n_slices + 1) * i - 1) for i in range(1, (n_slices + 1))]
    titles = ['Slice %d/%d' % (i, depth) for i in z]
    quick_imshow(
        nrows, ncols,
        [collage[:, :, z[i]] for i in range(n_slices)],
        titles=titles,
        figtitle=figtitle,
        figsize=figsize,
        vmax=vmax, vmin=vmin,
        colorbar=colorbar, tight=tight)
    if len(saveas) > 0:
        plt.savefig(saveas)
        plt.close()


def quick_plot(x_data, y_data=None, fmt='', color=None, xlim=None, ylim=None,
               label='', legends=False, x_label='', y_label='', figtitle='', annotation=None, figsize=(20, 10),
               f=None, ax=None, saveas=''):
    if f is None or ax is None:
        f, ax = subplots(figsize=figsize)
    if y_data is None:
        temp = x_data
        x_data = list(range(len(temp)))
        y_data = temp
    ax.plot(x_data, y_data, fmt, label=label, color=color)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if annotation is not None:
        for i in range(len(x_data)):
            annotate(annotation[i], (x_data[i], y_data[i]),
                     textcoords='offset points', xytext=(0, 10), ha='center')
    if len(x_label) > 0:
        ax.set_xlabel(x_label)
    if len(y_label) > 0:
        ax.set_ylabel(y_label)
    if len(figtitle) > 0:
        f.suptitle(figtitle)
    if legends:
        ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    ax.grid()
    if len(saveas) > 0:
        f.savefig(saveas, bbox_inches='tight')
    ax.grid()
    return f, ax


def quick_scatter(x_data, y_data=None, xlim=None, ylim=None,
                  label='', legends=False, x_label='', y_label='', figtitle='', annotation=None,
                  f=None, ax=None, saveas=''):
    if f is None or ax is None:
        f, ax = subplots()
    if y_data is None:
        temp = x_data
        x_data = list(range(len(temp)))
        y_data = temp
    ax.scatter(x_data, y_data, label=label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if annotation is not None:
        for i in range(len(x_data)):
            annotate(annotation[i], (x_data[i], y_data[i]),
                     textcoords='offset points', xytext=(0, 10), ha='center')
    if len(x_label) > 0:
        ax.set_xlabel(x_label)
    if len(y_label) > 0:
        ax.set_ylabel(y_label)
    if len(figtitle) > 0:
        f.suptitle(figtitle)
    if legends:
        ax.legend()
    ax.grid()
    if len(saveas) > 0:
        f.savefig(saveas)
    return f, ax


def quick_load(file_path, fits_field=1):
    if file_path.endswith('npz'):
        with load(file_path, allow_pickle=True) as f:
            data = f['arr_0']
            # Take care of the case where a dictionary is saved in npz format
            if isinstance(data, ndarray) and data.dtype == 'O':
                data = data.flatten()[0]
    # elif file_path.endswith(('pyc', 'pickle')):
    #     data = pickle_load(file_path)
    # elif file_path.endswith('fits.gz'):
    #     data = read_fits_data(file_path, fits_field)
    # elif file_path.endswith('h5'):
    #     data = read_hdf5_data(file_path)
    else:
        raise NotImplementedError(
            "Only npz, pyc, h5 and fits.gz are supported!")
    return data


def quick_save(file_path, data):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # For better disk utilization and compatibility with fits, use int32
    if file_path.endswith('npz'):
        savez_compressed(file_path, data)
    # elif file_path.endswith(('pyc', 'pickle')):
    #     save_object(file_path, data)
    # elif file_path.endswith('fits.gz'):
    #     if isinstance(data, ndarray) and data.dtype == int:
    #         data = data.astype(int32)
    #     save_fits_data(file_path, data)
    # elif file_path.endswith('h5'):
    #     write_hdf5_data(file_path, data)
    else:
        raise NotImplementedError(
            "Only npz, pyc, h5 and fits.gz are supported!")


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.
    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.
    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.
    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but '
                        f'got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def pad_nd_image(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit. by Fabian Isensee
    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by,
                          (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i])
                 for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [
                shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in
             range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + \
        list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer
