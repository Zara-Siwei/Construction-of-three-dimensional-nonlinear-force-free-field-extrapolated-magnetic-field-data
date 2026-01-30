# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:52:17 2025

@author: ano86532
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:14:40 2025

@author: andyl
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm,trange


# Path_hmi = 'D:/table/科研/磁计算/预报事件分析/sharp-events/10856/'
# filebx = Path_hmi + 'hmi.sharp_cea_720s.10856.20240222_220000_TAI.Bp.fits'
# fileby = Path_hmi + 'hmi.sharp_cea_720s.10856.20240222_220000_TAI.Bt.fits'
# filebz = Path_hmi + 'hmi.sharp_cea_720s.10856.20240222_220000_TAI.Br.fits'
# import pyfits as py
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm, trange
import os
import cv2

def safe_divide(a, b, fill_value=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result[np.isinf(result)] = fill_value
            result[np.isnan(result)] = fill_value
        elif np.isinf(result) or np.isnan(result):
            result = fill_value
    return result

Path0 = 'H:/new_dataset/'
savepath0 = 'G:/mag/'
noaas = os.listdir(Path0)  
noaas = sorted(noaas, key=lambda x: int(x))
# noaas = list(reversed(noaas))
for noaa in tqdm(noaas):
    
    if int(noaa) <= 11675:
        continue
    
    files = os.listdir(Path0+f'{noaa}/')  
    selected_files = [file for file in files if file.endswith('Br.fits')]
    harppp = selected_files[0].split('.')[2]
    timeee = selected_files[0].split('.')[3]
    for file in tqdm(selected_files,desc=f'now processing NOAANUM {noaa} HARPNUM {harppp} TIME {timeee[:8]}'):
        filebx = Path0 + f'{noaa}/'+file[:-7]+'Bp.fits'
        fileby = Path0 + f'{noaa}/'+file[:-7]+'Bt.fits'
        filebz = Path0 + f'{noaa}/'+file[:-7]+'Br.fits'
        
        harp = file.split('.')[2]
        
        savepath = savepath0+f'{harp}/'
        outputfile = savepath+file[:-7]+'nonlinearfff.fits'
        if os.path.exists(outputfile):
            continue
        
        try:
            # 检查NaN比例
            hdu = fits.open(filebx)
            bz0_array = np.array(hdu[1].data)
            nan_count = np.isnan(bz0_array).sum()
            total_elements = bz0_array.size
            nan_ratio = nan_count / total_elements
            if nan_ratio > 0.3:  # 如果NaN超过30%，跳过这个文件
                print("nan ratio > 30")
                continue
            
            
            # 加载并处理数据，立即处理NaN
            hdu = fits.open(filebx)
            bx0 = np.nan_to_num(hdu[1].data, nan=0.0)
            hdu = fits.open(fileby)
            by0 = np.nan_to_num(hdu[1].data, nan=0.0)
            hdu = fits.open(filebz)
            bz0 = np.nan_to_num(hdu[1].data, nan=0.0)
            

            
            cdelt = hdu[1].header['CDELT1']
            
            # 重设大小
            new_width = int(bz0.shape[1]*cdelt//(10*0.03))
            new_height = int(bz0.shape[0]*cdelt//(10*0.03))
            
            bx0 = cv2.resize(bx0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            by0 = cv2.resize(by0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            bz0 = cv2.resize(bz0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # 转置
            bx0 = bx0.transpose((1,0))
            by0 = by0.transpose((1,0))
            bz0 = bz0.transpose((1,0))
            
            dtype = bx0.dtype
            rsize, tsize = bx0.shape
            vsize = np.min([rsize, tsize])
            
            bx = np.zeros((rsize,tsize,vsize), dtype=dtype)
            by = np.zeros((rsize,tsize,vsize), dtype=dtype)
            bz = np.zeros((rsize,tsize,vsize), dtype=dtype)
            
            bx[:,:,0] = bx0
            by[:,:,0] = by0
            bz[:,:,0] = bz0
            
            # 初始化条件
            potlin = 1
            max_rst = np.max([rsize, tsize, vsize])
            default_dx = 2.0/(max_rst-1)
            
            x = np.empty(rsize, dtype=dtype)
            y = np.empty(tsize, dtype=dtype)
            z = np.empty(vsize, dtype=dtype)
            
            offset = -default_dx*float(rsize/2)
            x[:] = default_dx*np.arange(rsize)+offset
            offset = -default_dx*float(tsize/2)
            y[:] = default_dx*np.arange(tsize)+offset
            z[:] = default_dx*np.arange(vsize)
            
            dx = np.roll(x,-1)-x
            dx[rsize-1] = dx[rsize-2]
            dy = np.roll(y,-1)-y
            dy[tsize-1] = dy[tsize-2]
            dz = np.roll(z,-1)-z
            dz[vsize-1] = dz[vsize-2]
            wf = np.ones_like(bx)
            
            bzmax = np.max(bz[:,:,0])
            if bzmax > 0:
                bx = bx / bzmax
                by = by / bzmax
                bz = bz / bzmax
                
            alpha = 0.
            
            if potlin == 1:
                bydx = np.gradient(by[:,:,0], x, axis=0)
                bxdy = np.gradient(bx[:,:,0], y, axis=1)
                tmp_bz = bz[:,:,0].copy()
                xxx = np.where(np.abs(tmp_bz) < 1e-10)
                if len(xxx[0]) > 0:
                    tmp_bz[xxx] = 1.0e20
                loc_alph = safe_divide(bydx-bxdy, tmp_bz)
                if len(xxx[0]) > 0:
                    loc_alph[xxx] = 0.0
                sum_bz = np.sum(np.abs(bz[:,:,0]))
                if sum_bz > 0:
                    alpha = np.sum(loc_alph * np.abs(bz[:,:,0])) / sum_bz
                else:
                    alpha = 0
            
            # 初始化数组
            pbx = np.zeros((rsize,tsize,vsize), dtype=dtype)
            pby = np.zeros((rsize,tsize,vsize), dtype=dtype)
            pbz = np.zeros((rsize,tsize,vsize), dtype=dtype)
            
            # 计算系数
            coeffx = np.concatenate(([dx[0]], dx[0:rsize-2] + dx[1:rsize-1], [dx[rsize-2]])) * 0.5
            coeffy = np.concatenate(([dy[0]], dy[0:tsize-2] + dy[1:tsize-1], [dy[tsize-2]])) * 0.5
            
            coeffxp = np.tile(coeffx[:,None,None], (1,tsize,vsize))
            coeffyp = np.tile(coeffy[None,:,None], (rsize,1,vsize))
            
            # 准备坐标网格
            xp = np.tile(x[:,None,None], (1,tsize,vsize))
            yp = np.tile(y[None,:,None], (rsize,1,vsize))
            zp = np.tile(z[None,None,:], (rsize,tsize,1))
            
            cos_az = np.cos(alpha * zp)
            sin_az = np.sin(alpha * zp)
            bzp = np.tile(bz[:,:,0][:,:,None], (1,1,vsize))
            
            # 主要计算循环
            for j in trange(tsize):
                for i in range(rsize):
                    bigR = np.sqrt((x[i]-xp)**2 + (y[j]-yp)**2)
                    bigR = np.maximum(bigR, 1.0e-20)  # 避免除以零
                    r = np.sqrt(bigR**2 + zp**2)
                    r = np.maximum(r, 1.0e-20)  # 避免除以零
                    
                    cos_ar = np.cos(alpha * r)
                    sin_ar = np.sin(alpha * r)
                    
                    gamma = safe_divide(zp*cos_ar, bigR*r) - safe_divide(cos_az, bigR)
                    dgammadz = cos_ar*(1.0/(bigR*r)-zp**2/(bigR*r**3))-alpha*zp**2*sin_ar/(bigR*r**2)+alpha*sin_az/bigR
                    
                    gx = bzp * ((x[i] - xp) * dgammadz / bigR + alpha * gamma * (y[j] - yp) / bigR)
                    gy = bzp * ((y[j] - yp) * dgammadz / bigR - alpha * gamma * (x[i] - xp) / bigR)
                    gz = bzp * (zp*cos_ar/r**3 + alpha*zp*sin_ar/r**2)
                    
                    gx[i,j,:] = 0.0
                    gy[i,j,:] = 0.0
                    gz[i,j,:] = 0.0
                    
                    pbx[i,j,:] = np.sum(np.sum(coeffxp * coeffyp * gx, axis=1), axis=0) / np.pi/2.
                    pby[i,j,:] = np.sum(np.sum(coeffxp * coeffyp * gy, axis=1), axis=0) / np.pi/2.
                    pbz[i,j,:] = np.sum(np.sum(coeffxp * coeffyp * gz, axis=1), axis=0) / np.pi/2.
            
            bx[:,:,1:vsize] = pbx[:,:,1:vsize]
            by[:,:,1:vsize] = pby[:,:,1:vsize]
            bz[:,:,1:vsize] = pbz[:,:,1:vsize]
            
            # 保存结果
            pbxyz = np.stack((bx,by,bz))
            hdu = fits.PrimaryHDU(pbxyz)
            hdulist = fits.HDUList([hdu])
            savepath = savepath0 + f'{harp}/'
            os.makedirs(savepath, exist_ok=True)
            hdulist.writeto(savepath+file[:-7]+'linearfff.fits', overwrite=True)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue



        
        filebxyz=savepath+file[:-7]+'linearfff.fits'
        
        
        hdu=fits.open(filebxyz)
        data=hdu[0].data
        bx=data[0,:,:,:]
        by=data[1,:,:,:]
        bz=data[2,:,:,:]
        
        # hdu=py.open(filebxyz)
        # data=hdu[0].data
        # bx=data[0,:,:,:]
        # by=data[1,:,:,:]
        # bz=data[2,:,:,:]
        
        
        # bx=bx.transpose(2,1,0)
        # by=by.transpose(2,1,0)
        # bz=bz.transpose(2,1,0)
        
    
        ###############################################################
        # Initialization condition
        bc_flag=0
        potlin=1
        iterations = 10000
        dt = np.float64(0.00001)
        
        afd = 1.0e-6
        qsphere = 0
        abs_frac_diff=1.0e-6
        
        rsize,tsize,vsize=bx.shape
        max_rst = np.max([rsize, tsize, vsize])
        default_dx = 2.0/(max_rst-1)
        #####################################################################
        dtype=bx.dtype
        x = np.empty(rsize, dtype=dtype)
        y = np.empty(tsize, dtype=dtype)
        z = np.empty(vsize, dtype=dtype)
        
        offset = -default_dx*float(rsize/2)
        x[:]= default_dx*np.arange(rsize)+offset
        offset = -default_dx*float(tsize/2)
        y[:]= default_dx*np.arange(tsize)+offset
        z[:]= default_dx*np.arange(vsize)
        rsize1 = rsize-1
        tsize1 = tsize-1
        vsize1 = vsize-1
        
        dx=np.roll(x,-1)-x
        dx[rsize-1]=dx[rsize-2]
        dy=np.roll(y,-1)-y
        dy[tsize-1]=dy[tsize-2]
        dz=np.roll(z,-1)-z
        dz[vsize-1]=dz[vsize-2]
        wf=bx*0.+1.
        #####################################################################
        
        
        
        def deriv_dx(x,f):
            n=(x.shape)[0]
            n2=n-2
            nf=f.ndim
            nx, ny, nz = f.shape
            if nf !=3:
                print('F Array must be 3-d')
                return -1
            if (n < 3):
                print('Parameters must have at least 3 points')
                return -1
            if (n != nx):
                print('Mismatch between x and f[*,0,0]')
                return -1
            x12 = x-np.roll(x, -1)  #x0 - x1
            x01 = np.roll(x, 1)- x
            x02= np.roll(x, 1)-np.roll(x, -1)
            cx10 = (x12 / (x01 * x02))
            cx20 = (1. / x12 - 1. / x01)
            cx30 = (x01 / (x02 * x12))
            sx1 = (x01[1] + x02[1]) / (x01[1] * x02[1])
            sx2 = x02[1] / (x01[1] * x12[1])
            sx3 = x01[1] / (x02[1] * x12[1])
            rx1 = x12[n2] / (x01[n2] * x02[n2])
            rx2 = x02[n2] / (x01[n2] * x12[n2])
            rx3 = (x02[n2] + x12[n2]) / (x02[n2] * x12[n2])
            cx1 = np.empty((nx, ny, nz), dtype=dtype)
            cx2 = np.empty((nx, ny, nz), dtype=dtype)
            cx3 = np.empty((nx, ny, nz), dtype=dtype)
            for j in range(ny):
                for k in range(nz):
                    cx1[:, j, k] = cx10
                    cx2[:, j, k] = cx20
                    cx3[:, j, k] = cx30
            d=f*0.
            d = np.roll(f, 1,axis=0) * cx1+f * cx2 - np.roll(f, -1, axis=0) * cx3
            d[0, :, :] = f[0, :, :] *sx1 - f[1, :, :] *sx2 + f[2, :, :] *sx3
            d[n - 1, :, :] = -f[n - 3, :, :] *rx1 + f[n - 2, :, :] *rx2 - f[n - 1, :, :] *rx3
            return d
        def deriv_dy(y,f):
            n = (y.shape)[0]
            n2 = n - 2
            nf = f.ndim
            nx, ny, nz = f.shape
            if nf != 3:
                print('F Array must be 3-d')
                return -1
            if (n < 3):
                print('Parameters must have at least 3 points')
                return -1
            if (n != ny):
                print('Mismatch between x and f[0,:,0]')
                return -1
            y12 = y-np.roll(y, -1)  #x0 - x1
            y01 = np.roll(y, 1)- y
            y02= np.roll(y, 1)-np.roll(y, -1)
            cy10 = (y12 / (y01 * y02))
            cy20 = (1. / y12 - 1. / y01)
            cy30 = (y01 / (y02 * y12))
            sy1 = (y01[1] + y02[1]) / (y01[1] * y02[1])
            sy2 = y02[1] / (y01[1] * y12[1])
            sy3 = (y01[1] / (y02[1] * y12[1]))
            ry1 = y12[n2] / (y01[n2] * y02[n2])
            ry2 = y02[n2] / (y01[n2] * y12[n2])
            ry3 = (y02[n2] + y12[n2]) / (y02[n2] * y12[n2])
            cy1 = np.empty((ny, nx, nz), dtype=dtype)
            cy2 = np.empty((ny, nx, nz), dtype=dtype)
            cy3 = np.empty((ny, nx, nz), dtype=dtype)
            for j in range(nx):
                for k in range(nz):
                    cy1[:, j, k] = cy10
                    cy2[:, j, k] = cy20
                    cy3[:, j, k] = cy30
            cy1 = cy1.transpose(1, 0, 2)
            cy2 = cy2.transpose(1, 0, 2)
            cy3 = cy3.transpose(1, 0, 2)
            d = f * 0.
            d = np.roll(f, 1, axis=1) * cy1 + f * cy2 - np.roll(f, -1, axis=1) * cy3
            d[:, 0, :] = f[:, 0, :] *sy1 - f[:, 1, :] *sy2 + f[:, 2, :] *sy3
            d[:, n - 1, :] = -f[:, n - 3, :] *ry1 + f[:, n - 2, :] *ry2 - f[:, n - 1, :] *ry3
            return d
        def deriv_dz(z,f):
            n = (z.shape)[0]
            n2 = n - 2
            nf = f.ndim
            nx, ny, nz = f.shape
            if nf != 3:
                print('F Array must be 3-d')
                return -1
            if (n < 3):
                print('Parameters must have at least 3 points')
                return -1
            if (n != nz):
                print('Mismatch between x and f[0,0,:]')
                return -1
            z12 = z-np.roll(z, -1)  #x0 - x1
            z01 = np.roll(z, 1)- z
            z02= np.roll(z, 1)-np.roll(z, -1)
            cz10 = (z12 / (z01*z02))
            cz20 = (1./z12 - 1./z01)
            cz30 = (z01 / (z02 * z12))
            sz1 = (z01[1]+z02[1])/(z01[1]*z02[1])
            sz2 = z02[1]/(z01[1]*z12[1])
            sz3 = z01[1]/(z02[1]*z12[1])
            rz1 = z12[n2]/(z01[n2]*z02[n2])
            rz2 = z02[n2]/(z01[n2]*z12[n2])
            rz3 = (z02[n2]+z12[n2]) / (z02[n2]*z12[n2])
            cz1 = np.empty((nz, ny, nx), dtype=dtype)
            cz2 = np.empty((nz, ny, nx), dtype=dtype)
            cz3 = np.empty((nz, ny, nx), dtype=dtype)
            for j in range(ny):
                for k in range(nx):
                    cz1[:, j, k] = cz10
                    cz2[:, j, k] = cz20
                    cz3[:, j, k] = cz30
            cz1 = cz1.transpose(2, 1, 0)
            cz2 = cz2.transpose(2, 1, 0)
            cz3 = cz3.transpose(2, 1, 0)
            d = np.roll(f, 1, axis=2) * cz1 + f * cz2 - np.roll(f, -1, axis=2) * cz3
            d[:, :, 0] = f[:, :, 0] *sz1 - f[:, :, 1] *sz2 + f[:, :, 2] *sz3
            d[:, :, n - 1] = -f[:, :, n - 3] *rz1 + f[:, :, n - 2] *rz2 - f[:, :, n - 1] *rz3
            return d
        
        def curl_xyz(ax,ay,az,x,y,z):
            curl_x = deriv_dy(y, az) - deriv_dz(z, ay)
            curl_y = deriv_dz(z, ax) - deriv_dx(x, az)
            curl_z = deriv_dx(x, ay) - deriv_dy(y, ax)
            return curl_x,curl_y,curl_z
        def div_xyz(ax,ay,az,x,y,z):
            daxdx = deriv_dx(x, ax)
            daydy = deriv_dy(y, ay)
            dazdz = deriv_dz(z, az)
            div_a = daxdx + daydy + dazdz
            return div_a
        def cross_xyz(ax,ay,az,bx,by,bz):
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            return cx,cy,cz
        
        def vector_ops_fff(wf,x,y,z,bx,by,bz,):
            curl_bx,curl_by,curl_bz=curl_xyz(bx,by,bz,x,y,z)
            div_b=div_xyz(bx,by,bz,x,y,z)
            omega_x,omega_y,omega_z=cross_xyz(curl_bx,curl_by,curl_bz,bx,by,bz)
            b2 = (bx ** 2 + by ** 2 + bz ** 2)
            omega2 = (omega_x ** 2 + omega_y ** 2 + omega_z ** 2) ###add myself
            ok = np.where(b2 > 0.0)
            if ok == 0:
                print('No nonzero B')
                return curl_bx,curl_by,curl_bz,div_b,omega_x,omega_y,omega_z,b2
            omega_x[ok] = (omega_x[ok] - div_b[ok] * bx[ok]) / b2[ok]
            omega_y[ok] = (omega_y[ok] - div_b[ok] * by[ok]) / b2[ok]
            omega_z[ok] = (omega_z[ok] - div_b[ok] * bz[ok]) / b2[ok]
            omega2 = (omega_x ** 2 + omega_y ** 2 + omega_z ** 2) * wf
            omega_x = omega_x * wf
            omega_y = omega_y * wf
            omega_z = omega_z * wf
            return curl_bx,curl_by,curl_bz,div_b,omega_x,omega_y,omega_z,b2,omega2
        
        def obj_funct_fff(bx, by, bz, rsize, tsize, vsize, rsize1, tsize1, vsize1, b2, omega2, dx, dy, dz):
            a = np.array(dx[0])
            b = dx[0:rsize - 2] + dx[1:rsize - 1]
            c = np.array(dx[rsize - 2])
            # coeffx =np.concatenate((a,b,c),axis=0)
            a = np.append(a, b)
            a = np.append(a, c)
            coeffx = a * 0.5
        
            a = np.array(dy[0])
            b = dy[0:tsize - 2] + dy[1:tsize - 1]
            c = np.array(dy[tsize - 2])
            # coeffx =np.concatenate((a,b,c),axis=0)
            a = np.append(a, b)
            a = np.append(a, c)
            coeffy = a * 0.5
        
            a = np.array(dz[0])
            b = dz[0:vsize - 2] + dz[1:vsize - 1]
            c = np.array(dz[vsize - 2])
            # coeffx =np.concatenate((a,b,c),axis=0)
            a = np.append(a, b)
            a = np.append(a, c)
            coeffz = a * 0.5
        
            coeffxp3 = np.empty((rsize,tsize,vsize),dtype=dtype)
            coeffyp3 = np.empty((tsize,vsize,rsize),dtype=dtype)
            coeffzp3 = np.empty((vsize,tsize,rsize),dtype=dtype)
        
            for i in range(vsize):
                for j in range(tsize):
                    coeffxp3[:, j, i] = coeffx
            for i in range(rsize):
                for j in range(vsize):
                    coeffyp3[:, j, i] = coeffy
            for i in range(rsize):
               for j in range(tsize):
                   coeffzp3[:, j, i] = coeffz
            coeffxp3 = coeffxp3
            coeffyp3 = coeffyp3.transpose((2, 0, 1))
            coeffzp3 = coeffzp3.transpose((2, 1, 0))
            dV = (coeffxp3) * (coeffyp3) * (coeffzp3)
            ok = np.where(b2 > 0.0)
            if ok == 0:
                print('No nonzero B')
                return -1
            a=dV * b2 * omega2
            l= np.sum(dV * b2 * omega2)
            return l
        
        def force_fff(omega_x, omega_y, omega_z, curl_bx, curl_by, curl_bz,div_b,omega2,b2,bx, by, bz,x,y,z):
            #First curl of omega cross b
            temp_x, temp_y, temp_z = cross_xyz(omega_x, omega_y, omega_z, bx, by, bz)
            f_x, f_y, f_z = curl_xyz(temp_x, temp_y, temp_z, x, y, z)
            #Next subtract omega cross curl B
            temp_x, temp_y, temp_z = cross_xyz(omega_x, omega_y, omega_z, curl_bx, curl_by, curl_bz)
            f_x = f_x - (temp_x)
            f_y = f_y - (temp_y)
            f_z = f_z - (temp_z)
            #Next subtract grad omega dot B
            #temp = dot_xyz(omega_x, omega_y, omega_z, bx, by, bz)
            temp = omega_x*bx+omega_y*by+omega_z*bz
            #grad_xyz, temp, x, y, z, temp_x, temp_y, temp_z
            temp_x = deriv_dx(x, temp)
            temp_y = deriv_dy(y, temp)
            temp_z = deriv_dz(z, temp)
            f_x = f_x - (temp_x)
            f_y = f_y - (temp_y)
            f_z = f_z - (temp_z)
            #Add omega times div_b
            f_x = f_x + omega_x*div_b
            f_y = f_y + omega_y*div_b
            f_z = f_z + omega_z*div_b
            #Add omega^2 times B
            f_x = f_x + omega2*bx
            f_y = f_y + omega2*by
            f_z = f_z + omega2*bz
            return f_x,f_y,f_z
        
        def evolve_fff(i,wf,x,y,z,bx,by,bz,curl_bx, curl_by, curl_bz,div_b,omega_x, omega_y, omega_z,b2,omega2,rsize,tsize,vsize,dx, dy, dz,dt,l_value):
            afd = 1.0e-6
            rsize2 = rsize-1
            tsize2 = tsize-1
            vsize2 = vsize-1
        
            tbx= bx.copy()
            tby= by.copy()
            tbz= bz.copy()
        
            dbdt_x, dbdt_y, dbdt_z=force_fff(omega_x, omega_y, omega_z, curl_bx, curl_by, curl_bz,div_b,omega2,b2,bx, by, bz,x,y,z)
            localdt = dt #ok
            done = 0.
        
            while((localdt > dt/10000.0) and (done ==0)):
                bx[1:rsize2, 1:tsize2, 1:vsize2] = bx[1:rsize2, 1:tsize2, 1:vsize2] + dbdt_x[1:rsize2, 1:tsize2, 1:vsize2] * localdt
                by[1:rsize2, 1:tsize2, 1:vsize2] = by[1:rsize2, 1:tsize2, 1:vsize2] + dbdt_y[1:rsize2, 1:tsize2, 1:vsize2] * localdt
                bz[1:rsize2, 1:tsize2, 1:vsize2] = bz[1:rsize2, 1:tsize2, 1:vsize2] + dbdt_z[1:rsize2, 1:tsize2, 1:vsize2] * localdt
                curl_bx, curl_by, curl_bz, div_b, omega_x, omega_y, omega_z, b2, omega2 = vector_ops_fff(wf, x, y, z, bx, by, bz)
                local_l = obj_funct_fff(bx, by, bz, rsize, tsize, vsize, rsize1, tsize1, vsize1, b2, omega2, dx, dy, dz)
        
                frac_diff = np.abs((local_l - l_value) / l_value)
                if(frac_diff < afd):
                    print('Converged; frac_diff = ', frac_diff)
                    done = 1.
                    convergence_flag = 1.
                    dt = localdt
                    return local_l,convergence_flag,bx,by,bz,curl_bx, curl_by, curl_bz,div_b,omega_x, omega_y, omega_z,b2,omega2,dt
        
                if (local_l < l_value):
                    done = 1.
                    localdt = localdt * 1.01
                if (local_l >= l_value):
                    bx = tbx.copy()
                    by = tby.copy()
                    bz = tbz.copy()
        
                    curl_bx, curl_by, curl_bz, div_b, omega_x, omega_y, omega_z, b2, omega2 = vector_ops_fff(wf, x, y, z, bx,by, bz)
        
                    if(localdt > dt/10000.0):
                        done = 0.
                        localdt = localdt / 2.0
                        print('Reducing dt to ' , localdt)
                    if (localdt <= dt/10000.0):
                        done = 1.
            if(localdt <= dt/10000.0):
                convergence_flag = 1.
                print('NOT converging')
            if (localdt > dt/10000.0):
                convergence_flag = 0.
            dt = localdt
            return local_l,convergence_flag,bx,by,bz,curl_bx, curl_by, curl_bz,div_b,omega_x, omega_y, omega_z,b2,omega2,dt
        
        convergence_flag = 0.
        for i in range(iterations):
            if (convergence_flag == 1):
                # print('It seems to have converged or')
                # print('is not decreasing for small time step')
                break
            if (i ==0 ):
                curl_bx,curl_by,curl_bz,div_b,omega_x,omega_y,omega_z,b2,omega2=vector_ops_fff(wf,x,y,z,bx,by,bz)
                l_value = obj_funct_fff(bx, by, bz, rsize, tsize, vsize, rsize1, tsize1, vsize1, b2, omega2, dx, dy, dz)
                delta_l = 1.0e20
        
            local_l, convergence_flag, bx, by, bz, curl_bx, curl_by, curl_bz, div_b, omega_x, omega_y, omega_z, b2, omega2, dt = evolve_fff(i,wf,x,y,z,bx,by,bz,curl_bx, curl_by, curl_bz,div_b,omega_x, omega_y, omega_z,b2,omega2,rsize,tsize,vsize,dx, dy, dz,dt,l_value)
            delta_l = (l_value - local_l) / l_value
            # print(i, local_l, l_value)
            if (convergence_flag ==0 and local_l < l_value):
                l_value = local_l
            i_final = i
        pbxyz=np.stack((bx,by,bz))
        hdu = fits.PrimaryHDU(pbxyz)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(savepath+file[:-7]+'nonlinearfff.fits',overwrite=True)
