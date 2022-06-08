from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .forms import NNForm
from .models import NeuralNetwork
from .LLVisualizer import LossLandscapeVisualizer

counter = 0


def home(request):
	global counter
	if request.method == 'POST':
		print("IF POST")
		form = NNForm(request.POST, request.FILES)
		if form.is_valid():
			model = form.save()
			string = "[[1,1,1], [1,{},1], [1,1,1]];".format(counter)
			counter += 1
			print(type(model.min_value))
			print(model.min_value)
			visualizer = LossLandscapeVisualizer(model.nn_model.url, model.x_test.url, model.y_test.url, model.min_value, model.max_value, model.number_of_values)
			# model.delete()
			result = visualizer.process()

			models = NeuralNetwork.objects.all()
			for model in models:
				model.delete()
			return render(request, 'DEMOAPP/output.html', {'x': result['x'], 'y': result['y'], 'z': result['z'], 'form': form})

		# uploaded_file = request.FILES['document']
		# fs = FileSystemStorage()
		# print(uploaded_file.name)
		# fs.save(uploaded_file.name, uploaded_file)

	print("ELSE")
	string = "[[1,1,1], [2,2,2], [1,1,1]];"
	form = NNForm()
	return render(request, 'DEMOAPP/output.html', {'z1': string, 'form': form})

