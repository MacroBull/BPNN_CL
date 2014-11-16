# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 21:24:52 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull

"""

import os
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

mType='sine'
#mType='sin2'


if mType == 'sine':
	dim = (10, 1)
	step = (1000, 1000)
	sigm = '-DUSE_TANH'

if mType == 'sin2':
	dim = (30, 1)
	step = (1000, 1000)
	lr = 0

cmd = 'build/nncl {} {} {} {} {} {}'.format(mType, dim[0], dim[1], step[0], step[1], sigm)
print(cmd)
_, n_out = os.popen2(cmd)
#_, n_out = os.popen2('ls')

##########init#################
print("Started")

l = ''

while not(l.startswith('----')):
	l = n_out.readline()
	print(l.strip())


l = n_out.readline()
cnt = 1

if mType == 'sine':
	ls = 9
	lt = 361

	xs = linspace(0,2*pi, ls)
	xt = linspace(0,2*pi, lt)
	yo = sin(xs)
	ye = sin(xt)
	ys = zeros(ls)
	yt = zeros(lt)
	es = [10.]
	et = [10.]

	ion()
	gs = gridspec.GridSpec(3, 1)
	sp0 = subplot(gs[0:2, 0], title = r"$target=sin(x)(sigmoid={},\ dim={})$".format("tanh", repr(dim)))
	plot(xs, yo, 'o', label="Samples", alpha = 0.6)
	lyt = plot(xt, yt, label = "Output")[0]
	xlim(0,2*pi)
	xticks(xs, [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
		r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$',
		r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'
		])
	legend(loc='best')

	sp1 = subplot(gs[2, 0], title = r"$Error$")
	les = semilogy(es, 'o-', label="Err/Samples", alpha = 0.6)[0]
	let = semilogy(et, 'o-', label="Err/Tests", alpha = 0.6)[0]
	ylim(1e-5,1e1)
	xlabel('Trained epochs / kilo')
	legend(loc='best')
	draw()
	pause(0.01)

if mType == 'sin2':

	ls = 11*11
	lt = 21*21

	x = ogrid[-10.:11:2]
	z11 = sin(x)/x
	z11[5] = 1

	x = ogrid[-10.:11.]
	z21 = sin(x)/x
	z21[10] = 1

	xs = [(x,y) for x in linspace(-10,10,11) for y in linspace(-10,10,11)]
	xt = [(x,y) for x in linspace(-10,10,21) for y in linspace(-10,10,21)]
	yo = (z11*z11.reshape((1,-1)).T).reshape(-1)
	ye = (z21*z21.reshape((1,-1)).T).reshape(-1)
	ys = zeros(ls)
	yt = zeros(lt)
	es = [10.]
	et = [10.]

	x, y = mgrid[-10.:11.,-10.:11.]

	ion()
#
#	fig = figure()
#	bx = fig.add_subplot(111, title = r"$target=\frac{sin(x_1)}{x_1}\cdot\frac{sin(x_2)}{x_2}$", projection='3d')
#	bx.plot_surface(x,y,ye.reshape(21,-1), rstride=1, cstride=1, cmap=cm.coolwarm,
#        linewidth=0.5, antialiased=True)

	gs = gridspec.GridSpec(3, 1)
	ax = subplot(gs[0:2, 0], title = r"$\frac{sin(x_1)}{x_1}\cdot\frac{sin(x_2)}{x_2}$", projection='3d')
	ax.plot_surface(x,y,ye.reshape(21,-1), rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0.5, antialiased=True)

	subplot(gs[2, 0], title = r"$Error$")
	les = semilogy(es, 'o-', label="Err/Samples", alpha = 0.6)[0]
	let = semilogy(et, 'o-', label="Err/Tests", alpha = 0.6)[0]
	ylim(1e-3,1e1)
	xlabel('Trained epochs / kilo')
	legend(loc='best')
	draw()
	show()
	pause(0.1)



##########data#################
try:
	while not(l.startswith('----')):
		data = l.split()
		t, epochs, lr, mr, es_cur= [float(s) for s in data[:5]]
		if len(data)>5:
			yt = array([float(s) for s in data[5:]])
		print '\t'.join([repr(t), repr(epochs), repr(lr), repr(es_cur)])

		es.append(es_cur)
		et.append(sum((ye-yt)**2) * 0.5)
		cnt +=1
	#	sp0.plot(xt, yt)
		les.set_xdata(range(cnt))
		let.set_xdata(range(cnt))
		les.set_ydata(es)
		let.set_ydata(et)
		xlim(0,cnt)


		if mType == 'sine':
			lyt.set_ydata(yt)

		if mType == 'sin2':
			ax.clear()
			ax.plot_surface(x,y,yt.reshape(21,-1), rstride=1, cstride=1, cmap=cm.coolwarm,
						linewidth=0.5, antialiased=True)

		draw()
		savefig("Figures/{}@{}_dim={}_err=({:.5f},{:.5f})_lr={:.4f}_mr={:.4f}.svg".format(mType,
				int(epochs), repr(dim), es[-1], et[-1], lr, mr),
				facecolor='w', edgecolor='w', trasparent=False)
		pause(0.01)

		l = n_out.readline()

except BaseException, e:
	print(e)
finally:
	l = n_out.readline()

	print('-'*20 + 'Recordings' + '-'*20)
	print(es)
	print(et)
	print('-'*20 + 'Done' + '-'*20)
	print(es[-1], et[-1])
	print(l)

	print(os.popen3('killall nncl')[2].read())

	show(block=True)
