from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import externals.renderer.neural_renderer.neural_renderer as nr


class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0], texture_channel=3):
        super(Renderer, self).__init__()

        # texture
        self.texture_channel = texture_channel
        if texture_channel>255:
            raise ValueError('Texture channel must be less than 256!')
        if texture_channel>3:
            background_color = numpy.asarray(background_color) 
            if background_color.size<texture_channel:
                background_color = numpy.hstack((background_color, 
                    numpy.zeros(texture_channel-background_color.size,)))

        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        
        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction,
            texture_channel=self.texture_channel)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color, texture_channel=self.texture_channel)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        device = vertices.device
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction,
            texture_channel=self.texture_channel)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K.to(device)
            if R is None:
                R = self.R.to(device)
            if t is None:
                t = self.t.to(device)
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs.to(device)
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9)
            
            # remove faces with vertices that have negative depth
            # this is super important as otherwise they will be projected wrongly to the image and mess up the rendering
            v_pos = vertices[..., -1] > 0
            f_pos = v_pos.gather(dim=1, index=faces.flatten(1,2).long()).view(faces.shape)
            f_neg = torch.all(f_pos.__invert__(), dim=-1)
            f_pos = torch.all(f_pos, dim=-1)         # ensure all vertices of the face are positive
            
            # select pure negative samples
            if (f_pos.sum(dim=1) < faces.shape[1]).all():
                assert (f_neg.sum(dim=-1) > 0).all(), 'not enough negatives found'
                idx_neg = torch.argmax(f_neg, dim=1)
                neg_samples = faces[range(faces.shape[0]), idx_neg]
                
                # fill invalid faces with pure negatives
                faces = torch.where(f_pos.__invert__().unsqueeze(2).expand(faces.shape), neg_samples.unsqueeze(1).expand(faces.shape), faces)

            # rasterization
            faces = nr.vertices_to_faces(vertices, faces)
            out = nr.rasterize_rgbad(
                faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
                self.background_color, texture_channel=self.texture_channel)
        return out['rgb'], out['depth'], out['alpha']



if __name__=='__main__':
    """Rendering a teapot without anti-aliasing."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch()
    # test rendering with arbitrary dimension of textures 
    textures = torch.cat((textures, textures), dim=5)
    vertices = vertices.cuda()
    faces = faces.cuda()
    textures = textures.cuda()

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at', texture_channel=6)
    renderer.image_size = 256
    renderer.anti_aliasing = False

    # render
    images, _, _ = renderer(vertices, faces, textures)
    images = images.detach().cpu().numpy()
    image = images[2]
    image = image.transpose((1, 2, 0))

    imsave(os.path.join(data_dir, 'test_rasterize1.png'), image)
