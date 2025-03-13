import bpy
import os
import time
import random
import mathutils
import pickle
import numpy as np
# https://docs.blender.org/api/current/mathutils.geometry.html
from mathutils.geometry import barycentric_transform

from bpy.types import Panel  
from bl_ui.properties_render import RenderButtonsPanel  

bl_info = {
    "name": "Black Hole Render Engine",
    "bl_label": "BlackHoleRenderEngine",
    "blender": (4, 3, 2),
    "category": "Render",
}

################################################################################################################

########################################################
class BlackHoleRenderEngine(bpy.types.RenderEngine):
########################################################
    # Inspired by: https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
    bl_idname = "BlackHoleRenderEngine"
    bl_label = "BlackHole"
    bl_use_preview = True

    CP_DATA_INFO = "info"
    CP_DATA_GEODESICS = "geodesics"
    CP_DATA_BHHIT = "ray_blackhole_hit"
    CP_DATA_PIXELS = "pixel_coordinates"

    COLOR_BH = np.array([0,0,0,1]) # The BH is black!
    COLOR_HIT_TEST = np.array([0,1,0,1]) # Green can be used for testing
    COLOR_UNDEF_PIXEL = np.array([1,0,0,1]) # Red is used for bad behaving pixels
    COLOR_SOLID_COLOR_SKY = np.array([0,0,0,1]) # Used for the sky when debugging or when no image is supplied

    TEX_KEY_SKYDOME = "skydome_tex_name"
    #TEX_KEY_SPHERE = "sphere_tex_name"

    verbose = True
    v_debug = False

    # ------------------------------
    def render(self, depsgraph):
    # ------------------------------
        self.collision_detection = True
        self.object_texture_array_buffer = {}

        # Curvedpy data file
        pkl_file = bpy.path.abspath(depsgraph.scene.pkl_file)
        print(pkl_file)
        s_cp, curvedpy_data = self.loadAndCheckCurvedpyFile(pkl_file)
        if not s_cp: return

        self.flat_space = depsgraph.scene.flat_space

        # Skydome texture
        sky_image_path = bpy.path.abspath(depsgraph.scene.sky_image)
        s_tex, sky_tex_name = self.loadTextures(sky_image_path)
        #if not s_tex: return
        texture_names = {self.TEX_KEY_SKYDOME: sky_tex_name}

        # Sphere texture
        # sphere_image_path = bpy.path.abspath(depsgraph.scene.sphere_image)
        # s_tex, sphere_tex_name = self.loadTextures(sphere_image_path)
        # #if not s_tex: return
        # texture_names.update({self.TEX_KEY_SPHERE: sphere_tex_name})

        # Black hole location and object
        if depsgraph.scene.blackhole_obj == None:
            bh_loc = mathutils.Vector([0, 0, 0])
        else:
            bh_loc = depsgraph.scene.blackhole_obj.location

        # For some reason I need to update the depsgraph otherwise things wont render 
        # from the console properly using "-f <frame_nr>"
        # This might be a horrible hack :S
        depsgraph = bpy.context.evaluated_depsgraph_get() 
        depsgraph.update()

        # Start the rendering
        if self.is_preview:  # we might differentiate later
            pass             # for now ignore completely
        else:
            self.render_scene(depsgraph, curvedpy_data, texture_names, bh_loc)



    # ------------------------------
    def render_scene(self, depsgraph, curvedpy_data, texture_names, bh_loc):
    # ------------------------------
        res_y = curvedpy_data[self.CP_DATA_INFO]["height"]
        res_x = curvedpy_data[self.CP_DATA_INFO]["width"]

        buf = np.ones(res_x*res_y*4)
        buf.shape = res_y,res_x,4

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, res_x, res_y)
        layer = result.layers[0].passes["Combined"]

        # if self.pkl_file == "":
        #     print(f"=== Life rendering ===")
        #     for y in self.ray_trace(depsgraph, self.res_x, self.res_y, 1, buf, self.samples): 
        #         buf.shape = -1,4  
        #         layer.rect = buf.tolist()  
        #         self.update_result(result)  
        #         buf.shape = self.res_y, self.res_x,4  
        #         self.update_progress(y/self.res_y)  
        # else:
                
        buf = self.ray_trace(depsgraph, buf, curvedpy_data, texture_names, bh_loc)
        buf.shape = -1,4  
        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, res_x, res_y)
        layer = result.layers[0].passes["Combined"]

        layer.rect = buf.tolist()  

          
        self.end_result(result) 

    # ------------------------------
    def ray_trace(self, depsgraph, buf, curvedpy_data, texture_names, bh_loc):
    # ------------------------------
        height = curvedpy_data[self.CP_DATA_INFO]["height"]
        width = curvedpy_data[self.CP_DATA_INFO]["width"]



        # Buffer for the sample?
        sbuf = np.zeros(width*height*4)  
        sbuf.shape = height,width,4
    
        for i in range(len(curvedpy_data[self.CP_DATA_PIXELS])):
            if i%10000 == 0: print("Pixel progress: ", i, len(curvedpy_data[self.CP_DATA_PIXELS]))

            # If you need, get the camera coordinates:
            iy, ix, s = curvedpy_data[self.CP_DATA_PIXELS][i]

            # Do ray tracing over the geodesic:
            # Get the geodesic from curvedpy
            k_xyz, x_xyz = curvedpy_data[self.CP_DATA_GEODESICS][i]
            k_xyz, x_xyz = np.column_stack(k_xyz), np.column_stack(x_xyz)
            # Get if the BH was hit from curvedpy
            bh_hit = curvedpy_data[self.CP_DATA_BHHIT][iy, ix, s]

            # geodesic_cast gives a color or one geodesic. It checks if the geodesic hits
            # an object, the BH or goes into the background.
            #sbuf[iy, ix, 0:4] += (1/(s+1)) * self.geodesic_cast_debug(depsgraph, k_xyz, x_xyz, bh_hit, bh_loc, texture_names)
            if self.flat_space: # Act as if flat spacetime
                if "M" in curvedpy_data[self.CP_DATA_INFO].keys():
                    M = curvedpy_data[self.CP_DATA_INFO]["M"]
                else:
                    M = 1
                sbuf[iy, ix, 0:4] += (1/(s+1)) * self.straight_line_cast(depsgraph, k_xyz, x_xyz, bh_hit, bh_loc, texture_names, M)
            else: # Use geodesics
                sbuf[iy, ix, 0:4] += (1/(s+1)) * self.geodesic_cast(depsgraph, k_xyz, x_xyz, bh_hit, bh_loc, texture_names)


            #return
        
        #sbuf[:,:,0:3] = sbuf[:,:,0:3]/np.max(sbuf[:,:,0:3])
        buf = sbuf
        buf.shape = -1,4
        return buf

    # ------------------------------
    def geodesic_cast_debug(self, depsgraph, k_xyz, x_xyz, hit_blackhole, bh_loc, texture_names):
    # ------------------------------
        length = np.linalg.norm(x_xyz)
        # print(np.min(length), np.max(length), np.mean(length), np.std(length))
        # length = length/np.max(length)
        return np.array([length,length,length, 1])


    # ------------------------------
    def straight_line_cast(self, depsgraph, k_xyz, x_xyz, hit_blackhole, bh_loc, texture_names, M):
    # ------------------------------
        ray_origin, ray_direction = x_xyz[0], k_xyz[0]

        hit, loc_hit, normal_hit, index_hit, ob_hit, mat_hit = \
            depsgraph.scene.ray_cast(depsgraph, ray_origin, ray_direction)#, distance=distance)
        
        impact_vector = ray_origin - ray_origin.dot(ray_direction)*ray_direction
        # We save the length of the impact_vector. This is called the impact parameter in
        # scattering problems
        impact_par = np.linalg.norm(impact_vector)

        hit_blackhole = impact_par <= 2*M
        
        if hit:
            # In this case the geodesic hits an object in the scene
            hit_info = {"loc_hit":loc_hit, "normal_hit": normal_hit, "index_poly_hit":index_hit, \
                        "ob_hit":ob_hit, "mat_hit":mat_hit}

            print(np.array(ray_origin)-np.array(bh_loc))
            print(loc_hit)
            if hit_blackhole and (np.linalg.norm(np.array(ray_origin)-np.array(bh_loc))-2*M < np.linalg.norm(np.array(ray_origin)-np.array(loc_hit))): 
            # This is not exactly right but will be fine if things do not get close to the horizon
                return self.COLOR_BH
            else:
                return self.handleObjectHit(hit_info, texture_names)
        elif hit_blackhole:
            return self.COLOR_BH
        else:
            return self.handleBackgroundHit(k_xyz[0], texture_names)
        return self.COLOR_UNDEF_PIXEL

    # ------------------------------
    def geodesic_cast(self, depsgraph, k_xyz, x_xyz, hit_blackhole, bh_loc, texture_names):
    # ------------------------------
        # The geodesic coordinates are relative to the black hole location and need to be translated
        # bh_loc
        
        # We walk over every step of the geodesic and do a 'classic', straight ray_cast for every step
        # print("R", hit_blackhole, np.linalg.norm(x_xyz[-1]))
        for i in range(len(x_xyz)):
            if i == len(x_xyz)-1:
                # If you are at the last step of the geodesic, the geodesic did not hit any objects
                # Thus either it hits the blackhole or it goes to infinity (and beyond!)
                if hit_blackhole:
                    return self.COLOR_BH
                else:
                    # The only option now is that the geodesic did not hit the BH and did not hit an object
                    # Thus we color it with the background sky image
                    # if np.linalg.norm(k_xyz[i]) == 0:
                    #     _k = x_xyz[i] - x_xyz[i-1]
                    #     _k = _k/np.linalg.norm(_k)
                    # else:
                    _k = k_xyz[i]
                    return self.handleBackgroundHit(_k, texture_names)
            else: 
                # If it is not the last step of the geodesic, only ray_cast as far as the next step of the
                # geodesic
                distance = np.linalg.norm(x_xyz[i+1] - x_xyz[i])

            hit, loc_hit, normal_hit, index_hit, ob_hit, mat_hit = \
                depsgraph.scene.ray_cast(depsgraph, x_xyz[i], k_xyz[i], distance=distance)

            if hit:
                # In this case the geodesic hits an object in the scene
                hit_info = {"loc_hit":loc_hit, "normal_hit": normal_hit, "index_poly_hit":index_hit, \
                            "ob_hit":ob_hit, "mat_hit":mat_hit}

                return self.handleObjectHit(hit_info, texture_names)
                #return hit, 
                
        # If for some strange reason we do not get any colors, color it as undefined so we can debug
        return self.COLOR_UNDEF_PIXEL

    # ------------------------------
    def handleObjectHit(self, hit_info, texture_names):
    # ------------------------------
        emission = True # Use emission or use lighting

        # Try to retrieve the texture of the object
        # hit_info["mat_hit"].texture_slots[object_tex_name]

        if True: # If a texture is present on the object        
            if not emission: # Use lighting from lamps
                # intensity for all lamps  
                # eps: small offset to prevent self intersection for secondary rays
                
                # We collect the light objects in our scene            
                #lamps = [ob for ob in depsgraph.scene.objects if ob.type == 'LIGHT']  
                
                color = np.zeros(3)
                base_color = np.ones(3) * intensity  # light color is white  
                for lamp in lamps:  
                    # for every lamp determine the direction and distance  
                    light_vec = lamp.location - loc  
                    light_dist = light_vec.length_squared  
                    light_dir = light_vec.normalized()  
                     
                    # cast a ray in the direction of the light starting  
                    # at the original hit location  
                    lhit, lloc, lnormal, lindex, lob, lmat = depsgraph.scene.ray_cast(depsgraph, loc+light_dir*eps, light_dir)  
                     
                    # if we hit something we are in the shadow of the light  
                    if not lhit:  
                        # otherwise we add the distance attenuated intensity  
                        # we calculate diffuse reflectance with a pure   
                        # lambertian model  
                        # https://en.wikipedia.org/wiki/Lambertian_reflectance  
                        color += base_color * intensity * normal.dot(light_dir)/light_dist  

            else: #emission=True
                loc = hit_info['loc_hit']
                ob = hit_info['ob_hit']
                poly_index = hit_info['index_poly_hit']

                if True:
                    color = self.generalObjectHitColor(loc_hit_world=loc, index_poly_hit=poly_index, ob_hit=ob, verbose = False)
                else:
                    color = self.sphereObjectHitColor(loc, ob, texture_names)

            return color

        else: # If no texture is found on the object
            #print(hit_info["mat_hit"].texture_slots[object_tex_name]
            return self.COLOR_HIT_TEST

    # ------------------------------
    def generalObjectHitColor(self, loc_hit_world, index_poly_hit, ob_hit, verbose = False):
    # ------------------------------
    # Check: https://blender.stackexchange.com/questions/222327/getting-the-colour-information-from-texture-at-ray-intersection-using-bvhtree-re
    # And: https://blender.stackexchange.com/questions/79236/access-color-of-a-point-given-the-3d-position-on-the-surface-of-a-polygon

        # Check if there is a material and an image texture
        if len(ob_hit.material_slots) == 0: # Object has no material slot
            return self.COLOR_UNDEF_PIXEL

        if not ('Image Texture' in ob_hit.material_slots[0].material.node_tree.nodes.keys()):
            if 'Principled BSDF' in ob_hit.material_slots[0].material.node_tree.nodes.keys(): #Emission?
                return np.array(ob_hit.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value)
            else:
                return self.COLOR_UNDEF_PIXEL

        ob_hit.material_slots[0].material.node_tree.nodes['Image Texture']

        if verbose: print("LOC HIT WORLD: ", loc_hit_world)
        loc_hit = ob_hit.matrix_world.inverted() @ loc_hit_world
        if verbose: print("LOCATION HIT OBJ: ", loc_hit)
        
        # Get the vertices of the hit polygon
        verts_indices = ob_hit.data.polygons[index_poly_hit].vertices

        # Polygon vertices coordinates in 3D space
        p1, p2, p3 = [ob_hit.data.vertices[verts_indices[i]].co for i in range(3)]
        if verbose: print("POLYGON VERTICES: ", p1, p2, p3, p4)
        
        # Get the loop indices of the hit polygon
        uvMap_indices = ob_hit.data.polygons[index_poly_hit].loop_indices
        if verbose: print("POLYGON LOOP INDICES: ", uvMap_indices)

        # Get the UVMap of the object and get the coordinates using the loop indices
        uvMap = ob_hit.data.uv_layers['UVMap']
        if verbose: print("uvMap", uvMap)
        uv_1, uv_2, uv_3 = [uvMap.data[uvMap_indices[i]].uv for i in range(3)]
        if verbose: print("uv", uv_1, uv_2, uv_3)
        
        # The uv coordinates form a triangle. This adds a zero third component for
        # it to work with barycentric_transform
        uv_1 = uv_1.to_3d()
        uv_2 = uv_2.to_3d()
        uv_3 = uv_3.to_3d()
        if verbose: print("uv3d", uv_1, uv_2, uv_3)

        # We have the ray hit in object coordinates that is part of a polygon spanned
        # by (p1, p2, p3). We want the loc_hit in coordinates spanned by uv_1, uv_2, 
        # uv_3 and use barycentric_transform to get it
        # p1, p2, p3 is the source triangle => uv_1, uv_2, uv_3 is the source triangle:
        loc_hit_bar = barycentric_transform( loc_hit, p1, p2, p3, uv_1, uv_2, uv_3 )
        if verbose: print("loc_hit_bar", loc_hit_bar)
        # We need to resize it to 2D coordinates
        loc_hit_bar.resize_2d()
        if verbose: print("loc_hit_bar 2d", loc_hit_bar)
        
        # We get the image from the node tree
        if ob_hit.name in self.object_texture_array_buffer.keys():
            pixels = self.object_texture_array_buffer[ob_hit.name]
            height, width = pixels.shape[0],pixels.shape[1]
        else:
            image = ob_hit.material_slots[0].material.node_tree.nodes['Image Texture'].image
            # We get the pixels as array and get the width and height of the image to resize the
            # numpy array
            pixels = np.array( image.pixels ) #Faster than accessible image.pixels[x] each time
            width = image.size[0]
            height = image.size[1]
            pixels = np.reshape(pixels, (height,width,4))
            # Save it in the buffer for next hit
            self.object_texture_array_buffer.update({ob_hit.name: pixels})

        # Then we get the color from the image using our coordinates
        uv_x = round(loc_hit_bar[0]*(width-1))
        uv_y = round(loc_hit_bar[1]*(height-1))

        rgb = pixels[uv_y,uv_x,:]
        if verbose: print("RGB IMG: ", rgb*255, uv_y, uv_x)
        
        return rgb

    # # ------------------------------
    # def sphereObjectHitColor(self, loc, ob, texture_names):
    # # ------------------------------
    #     hit_norm = loc-ob.matrix_world.translation#ob.location
    #     hit_norm = hit_norm/np.linalg.norm(hit_norm)
    #     th = np.arccos(hit_norm[2])
    #     #ph = np.arctan(hit_norm[1]/hit_norm[0])
    #     ph = np.atan2(hit_norm[1],hit_norm[0])+np.pi

    #     #color=np.array([1, 0, 0])
    #     print(th, ph, loc, ob.location)
    #     color = np.array(
    #         [*bpy.data.textures[texture_names[self.TEX_KEY_SPHERE]].evaluate((ph/(2*np.pi),th/np.pi,0)).xyz, 1]
    #         )
    #     # color = np.array(
    #     # [*bpy.data.textures[texture_names[self.TEX_KEY_SKYDOME]].evaluate( (-phi,2*theta-1,0) ).xyz,1]
    #     # )
    #     return color

    # ------------------------------
    def handleBackgroundHit(self, direction, texture_names):
    # ------------------------------
        #color = np.zeros(3)  
        if texture_names[self.TEX_KEY_SKYDOME] in bpy.data.textures and texture_names[self.TEX_KEY_SKYDOME] != "":
            if direction[2] > 1 or direction[2]<-1:
                print("Wrong, dir not normalized: ", direction)
                direction = direction / np.linalg.norm(direction)

            theta = 1-np.arccos(direction[2])/np.pi
            phi = np.arctan2(direction[1], direction[0])/np.pi
            color = np.array(
                [*bpy.data.textures[texture_names[self.TEX_KEY_SKYDOME]].evaluate( (-phi,2*theta-1,0) ).xyz,1]
                )
        else:
            color = self.COLOR_SOLID_COLOR_SKY
        return color

    # ------------------------------
    def loadTextures(self, sky_image_path):
    # ------------------------------
        if not os.path.isfile(sky_image_path):
            print(f"=== FAIL: Texture file not found: {sky_image_path} ===")
            return False, ""

        # Skydome image and texture
        sky_image_filename = os.path.basename(sky_image_path)
        sky_tex_name = sky_image_filename + "_tex"

        if not sky_image_filename in bpy.data.images:
            print("LOADING:", sky_image_filename)
            bpy.data.images.load(sky_image_path)
        if not sky_tex_name in bpy.data.textures:
            sky_tex = bpy.data.textures.new(sky_tex_name, "IMAGE")
            sky_tex.image = bpy.data.images[sky_image_filename]
        else:
            sky_tex = bpy.data.textures[sky_tex_name]

        if self.verbose: print(f"== SUCCESS: Loaded Skydome image: {sky_image_filename} ==")

        return True, sky_tex_name

    # ------------------------------
    def loadAndCheckCurvedpyFile(self, pkl_file):
    # ------------------------------
        if pkl_file:
            loadSucces, dictOk, curvedpy_data = self.loadPickeFile(pkl_file)
            if loadSucces and dictOk: 
                if self.verbose: 
                    print(f"== SUCCESS: Loaded curvedpy data succesful ==")
                if self.v_debug: 
                    print(f"     Camera info: {curvedpy_data[self.CP_DATA_INFO]}")
                    print(f"     Cam file: {pkl_file}")
                return True, curvedpy_data

            elif loadSucces and not dictOk: 
                print(f"== FAIL: Loading of curvedpy data succesful, BUT keys in dict not OK, stopping ==")
                print(f"     Camera info: {curvedpy_data.keys()}")
                if self.v_debug: print(f"     Cam file: {pkl_file}")
                return False, curvedpy_data
            else:
                print(f"== FAIL: Loading of curvedpy data *NOT* succesful, stopping ==")
                return False, {}
        else:
            print(f"== FAIL: No curvedpy pkl file, stopping ==")
            return False, {}

    # ------------------------------
    def loadPickeFile(self, pkl_file):
    # ------------------------------
        if os.path.isfile(pkl_file):
            with open(pkl_file, 'rb') as f:
                filecontent = pickle.load(f)
                if self.v_debug: print(filecontent.keys())
                # info = filecontent["info"]
                # geodesics = filecontent["geodesics"]
                # #self.ray_end = filecontent["ray_end"]
                # ray_blackhole_hit = filecontent["ray_blackhole_hit"]
                # pixel_coordinates = filecontent["pixel_coordinates"]

                check = (self.CP_DATA_INFO in filecontent) and (self.CP_DATA_GEODESICS in filecontent) and \
                        (self.CP_DATA_BHHIT in filecontent) and (self.CP_DATA_PIXELS in filecontent)

                return True, check, filecontent

        else:
            if self.v_debug: print(f"CAM: FILE NOT FOUND {filepath}")
            return False, False, {}


########################################################
# END BlackHoleRenderEngine
########################################################



################################################################################################################
# Below are UI and Blender functions
################################################################################################################

########################################################
class CUSTOM_RENDER_PT_blackhole(RenderButtonsPanel, Panel):  
########################################################
    bl_label = "Blackhole Settings"  
    COMPAT_ENGINES = {BlackHoleRenderEngine.bl_idname}  
  
    def draw_header(self, context):  
        rd = bpy.context.scene
  
    def draw(self, context):  
        layout = self.layout  
  
        rd = bpy.context.scene#.render  
        #layout.active = rd.use_antialiasing  
  
        split = layout.split()  
  
        col = split.column()  
        col.row().prop(rd, "blackhole_obj", text="Blackhole")#, expand=True)  
        col.row().prop(rd, "flat_space", text="Flat space")#, expand=True)  
        col.row().prop(rd, "pkl_file", text="pkl_file")
        col.row().prop(rd, "sky_image", text="Sky image")
        #col.row().prop(rd, "sphere_image", text="Sphere image")



PROPS = [
     ('blackhole_obj', bpy.props.PointerProperty(name='blackhole_obj', type=bpy.types.Object)),
     ('flat_space', bpy.props.BoolProperty(name='flat_space', default=False)),
     ('pkl_file', bpy.props.StringProperty(name='pkl_file', default="", subtype="FILE_PATH")),
     ('sky_image', bpy.props.StringProperty(name='sky_image', default="", subtype="FILE_PATH")),
     #('sphere_image', bpy.props.StringProperty(name='sphere_image', default="", subtype="FILE_PATH")),
 ]


def get_panels():
     exclude_panels = {
         'VIEWLAYER_PT_filter',
         'VIEWLAYER_PT_layer_passes',
     }

     panels = []
     for panel in bpy.types.Panel.__subclasses__():
         if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
             if panel.__name__ not in exclude_panels:
                 panels.append(panel)

     return panels


def register():
    # Register the RenderEngine
    bpy.utils.register_class(BlackHoleRenderEngine)
    bpy.utils.register_class(CUSTOM_RENDER_PT_blackhole)

    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        panel.COMPAT_ENGINES.add(BlackHoleRenderEngine.bl_idname)#'gr_ray_tracer')#CUSTOM')
    
    from bl_ui import (
            properties_render,
            properties_material,
            properties_world,
            )

    from cycles import(ui)
    properties_world.WORLD_PT_context_world.COMPAT_ENGINES.add(BlackHoleRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_context_material.COMPAT_ENGINES.add(BlackHoleRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_surface.COMPAT_ENGINES.add(BlackHoleRenderEngine.bl_idname)


def unregister():
    bpy.utils.unregister_class(BlackHoleRenderEngine)
    bpy.utils.unregister_class(CUSTOM_RENDER_PT_blackhole)

    
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)
        
    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        if BlackHoleRenderEngine.bl_idname in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove(BlackHoleRenderEngine.bl_idname)

if __name__ == "__main__":
    register()


