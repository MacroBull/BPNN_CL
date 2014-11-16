float logistic(float x){
	return 1./(1.+exp(-x));
}

float dlogisticy(float y){
	return y - y*y;
}

float dtanhy(float y){
	return 1. - y*y;
}
