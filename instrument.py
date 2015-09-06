import os
import time
import numpy
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from math import sqrt
np=numpy

class usbtmc:
    """Simple implementation of a USBTMC device driver, in the style of visa.h"""

    def __init__(self, device):
        self.device = device
        try:
            self.FILE = os.open(device, os.O_RDWR)
        except OSError:
            raise OSError('Plug in scope and run ' +
                    '"sudo chmod a+rw /dev/usbtmc0"')

    def write(self, command):
        os.write(self.FILE, command);
        #time.sleep(.1)

    def read(self, length = 10000):
        return os.read(self.FILE, length)

    def getName(self):
        self.write("*IDN?")
        return self.read(300)

    def sendReset(self):
        self.write("*RST")


class AgilentScope:
    """Class to control a Rigol DS1000 series oscilloscope"""
    def __init__(self, device):
        self.meas = usbtmc(device)

        self.name = self.meas.getName()

        print "Connected to " + self.name

    def write(self, command):
        """Send an arbitrary command directly to the scope"""
        self.meas.write(command)

    def read(self, length=10000):
        """Read an arbitrary amount of data directly from the scope"""
        return self.meas.read(length)

    def get_channel_data(self, channel_number):
        """Automatically returns the y-values in volts for the specified channel"""
        self.write(":WAV:POIN:MODE RAW")
        self.write(":WAV:FORM BYTE")
        self.write(":WAVEFORM:POINTS 10240")
        self.write(":STOP")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        time.sleep(0.1)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        rawdata = self.read(number_data_points)
        #data = numpy.frombuffer(rawdata, 'B')
        data = numpy.frombuffer(rawdata,np.uint8)
        while len(data)<number_data_points:
            print  ("Data length should be %d, but is only %d" %
                    (number_data_points,len(data)))
            raise BufferError( "Data length should be %d, but is only %d" %
                (number_data_points,len(data)) )
        return data

    def check_channel_scaling(self,channel_number=1):
        """Gets first entry as an ASCII string and prints out conversion as well"""
        #Get and convert data usual way
        series=self.get_multiple_traces(n_traces=1,
                channel_number=channel_number)
        converted_data=series.converted_data[0,0]
        #Get data converted by scope to ascii string for comparison
        self.write("WAV:FORM ASCII")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        ascii_data= self.read(25).split(',')[0]
        #Print data for comparison
        print "ASCII: %s, and converted: %f" % (ascii_data, converted_data)
        #Unlock scope
        self.unlock()
        return

    def get_single_trace(self,channel_number):
        """Takes a new trace and returns the channel_data"""
        self.write(":SINGLE") #Set trigger to single
        self.write(":TRIGger:STATus?") #Wait until scope takes data
        trigger_status=self.read()
        while trigger_status!='STOP\n':
            time.sleep(0.01)
            self.write(":TRIGger:STATus?")
            trigger_status=self.read()
        try:
            data=self.get_channel_data(channel_number) #Retrieve scope data
        except BufferError:
            print "Trying to take another trace..."
            data=self.get_single_trace(channel_number)
        return data

    def get_multiple_traces(self,n_traces=10,channel_number=1):
        """Takes multiple traces and returns an array with each trace as a row"""
        #Prepare object to hold data
        series=Measurement_Series(self)
        series.get_scope_settings()
        #Start taking data
        one_data=self.get_single_trace(channel_number)
        all_data=numpy.zeros([n_traces,len(one_data)],dtype=np.uint8)
        all_data[0]=one_data
        for j in range(1,n_traces):
            one_data=self.get_single_trace(channel_number)
            all_data[j]=one_data
        series.channel_data=all_data
        self.unlock()
        return series

    def measure_tau(self,n_traces=10,channel_number=1,print_result=True):
        """Returns tau, tau_uncertainty, and a measurement_series instance"""
        series=self.get_multiple_traces(n_traces,channel_number)
        tau=series.tau_mean
        tau_uncertainty=series.tau_uncertainty
        if print_result:
            print "Tau is (%1.2f +/- %1.2f)us" % (tau*1e6,tau_uncertainty*1e6)
        return tau,tau_uncertainty,series

    def measure_repeatedly(self,n_traces=10,channel_number=1):
        """Constantly measures tau and prints the result

        n_traces gives the number of traces to fit/average to get tau"""
        while True:
            self.measure_tau(self,n_traces=n_traces,
                    channel_number=channel_numberm,
                    print_result=True)

    def unlock(self):
        """Unlocks the buttons on the scope for manual use"""
        self.write(":KEY:LOCK DISable")

    def reset(self):
        """Reset the instrument"""
        self.meas.sendReset()


class Measurement_Series(object):
    """Handles the data from one series of measurements"""

    #Class constants
    left_plot_limit=-1.5 #time in us
    right_plot_limit=20  #time in us
    initial_params=(0.06,2e-6,0.02) #Amplitude, decay time, offset
    left_fit_limit=0.7e-6 #time to begin fit
    right_fit_limit=15e-6 #time to end fit
    filter_order=10
    filter_cutoff=10e6 #in Hz

    def __init__(self,scope):
        """Initializes a Measurement_Series instance from an AgilentScope"""
        self.scope=scope
        self._channel_data=np.array([])
        self.time_offset=None
        self.delta_t=None
        self.y_reference=None
        self.y_origin=None
        self.y_increment=None
        self._n_traces=None
        self._trace_length=None
        self._did_fit=False
        self._params=np.array([])
        self.file_name=''
        self._butter_lowpass_polynomials=None

    def get_scope_settings(self):
        """Sets attributes to reflect current scope settings"""
        scope=self.scope
        scope.write(":WAV:XOR?")
        self.time_offset=float(scope.read(20))
        scope.write(":WAV:XINC?")
        self.delta_t=float(scope.read(20))
        scope.write(":WAV:YREF?")
        self.y_reference=float(scope.read(20))
        scope.write(":WAV:YOR?")
        self.y_origin=float(scope.read(20))
        scope.write(":WAV:YINC?")
        self.y_increment=float(scope.read(20))

    @property
    def channel_data(self):
        """The uint8 data from the scope, each row is one trace"""
        return self._channel_data
    @channel_data.setter
    def channel_data(self,value):
        self._channel_data=value
        self._n_traces=self.channel_data.shape[0]
        self._trace_length=self.channel_data.shape[1]
        self._did_fit=False

    @property
    def converted_data(self):
        """Channel_data converted to real voltages (in volts)"""
        channel_data=self.channel_data
        if channel_data.size==0:
            raise AttributeError('Series must have measurement data before '+
                    'trying to convert it')
        converted_data=((self.y_reference-1)-channel_data)*self.y_increment-self.y_origin
        return converted_data

    @property
    def filtered_data(self):
        """converted_data (in volts) passed through a Buttersworth filter"""
        converted_data=self.converted_data
        b, a = self.butter_lowpass_polynomials
        filtered_data = lfilter(b, a, converted_data)
        return filtered_data

    @property
    def time_data(self):
        """The times (in seconds) corresponding to the channel_data"""
        length=self.trace_length
        time_data=numpy.linspace(0.,length-1,length)
        time_data=time_data*self.delta_t+self.time_offset
        return time_data

    @property
    def n_traces(self):
        """The number of traces in this Measurement_Series instance"""
        return self._n_traces

    @property
    def trace_length(self):
        """The number of data points in each trace"""
        return self._trace_length

    @property
    def params(self):
        """Array with each row containing fit parameters"""
        if self._params.size==0:
            self.fit_data()
        return self._params
    @params.setter
    def params(self,value):
        self._params=value

    @property
    def tau_array(self):
        """Array of all tau values from fitting traces"""
        return self.params[:,1]

    @property
    def tau_mean(self):
        """The mean tau from all the fits in the Measurement_Series in sec"""
        return np.mean(self.tau_array)

    @property
    def tau_std(self):
        """The standard deviation of tau from the fits in sec"""
        return np.std(self.tau_array,ddof=1)

    @property
    def tau_uncertainty(self):
        """The error on tau_mean in sec"""
        return self.tau_std/sqrt(self.n_traces)

    def fit_data(self):
        """Fits fit_function to each trace and stores the results"""
        left_index=self.time_to_index(self.left_fit_limit)
        right_index=self.time_to_index(self.right_fit_limit)
        time_data=self.time_data[left_index:right_index]
        filtered_data=self.filtered_data
        initial_params=self.initial_params
        params_array=np.zeros([self.n_traces,len(self.initial_params)])
        j=0;
        for trace in filtered_data:
            one_params=curve_fit(fit_function, time_data,
                    trace[left_index:right_index], initial_params,
                    #Dfun=self.fit_jacobian)[0]
                    Dfun=None)[0]
            params_array[j]=one_params
            j+=1
        self.params=params_array
        self._did_fit=True

    def time_to_index(self,time):
        """Gives time_data index corresponding to time (in seconds)"""
        index=(time-self.time_offset)/self.delta_t
        index=round(index)
        index=min([index,self.trace_length-1])
        index=max([index,0])
        return index

    def copy_data(self,series):
        """Copies scope data from the given series

        This is helpful when updating this class while you're not near the
        scope so you can't take any data"""
        self.scope=series.scope
        self.time_offset=series.time_offset
        self.delta_t=series.delta_t
        self.y_reference=series.y_reference
        self.y_origin=series.y_origin
        self.y_increment=series.y_increment
        self.channel_data=series.channel_data

    @staticmethod
    def fit_function(t,A,tau,c):
        """Function for fitting data: Gaussian with constant offset"""
        return A*np.exp(-t/tau)+c

    @staticmethod
    def fit_jacobian(params,t,ydata,fit_function):
        """Jacobian of derivatives for fit function"""
        A,tau,c=params
        exp=np.exp(-t/tau)
        term1=exp.reshape(-1,1)
        term2=(A*exp/tau**2).reshape(-1,1)
        term3=np.ones((len(t),1))
        res = np.hstack( ( term1, term2, term3) )
        return res

    def plot_ringdown(self,trace_number=0,plot_unfiltered=True,
            plot_filtered=True,plot_fit=True):
        """Plots the ringdown and the fit"""
        plt.figure()
        tau=self.tau_array[trace_number]
        plt.title("Ringdown tau=%1.2fus %s" % (tau*1e6,self.file_name))
        plt.ylabel("Voltage (V)")
        plt.xlabel("time (us)")
        plt.xlim(self.left_plot_limit,self.right_plot_limit)
        time_data=self.time_data
        if plot_unfiltered:
            plt.plot(time_data*1e6,self.converted_data[trace_number],
                    color='k',label='data')
        if plot_filtered:
            plt.plot(time_data*1e6,self.filtered_data[trace_number],
                    color='g',label='filtered data')
        if plot_fit:
            params=self.params[trace_number]
            left_index=self.time_to_index(self.left_fit_limit)
            right_index=self.time_to_index(self.right_fit_limit)
            fit_times=time_data[left_index:right_index]
            fit_vals=self.fit_function(fit_times,*params)
            plt.plot(fit_times*1e6,fit_vals,color='r',label='fit')
        legend=plt.legend()
        #Make legend lines thicker and easier to see
        for obj in legend.legendHandles:
            obj.set_linewidth(3.0)

    def save(self,overwrite=False):
        """Saves measurement_series instance to disk

        Path is taken relative to the Ringdown_Data subdirectory"""
        if not self.file_name:
            raise AttributeError('self.file_name must be set before saving')
        file_name=os.path.join('Ringdown_Data',self.file_name)
        if overwrite==False and os.path.exists(file_name):
            raise NameError('File %s already exists. ' % file_name+
                    'Set overwrite=True to overwrite it')
        with open(file_name,'wb') as file:
            pickle.dump(self,file,2)

    @classmethod
    def load(cls,file_name):
        """Loads the file from disk and returns result"""
        file_name=os.path.join('Ringdown_Data',file_name)
        with open(file_name,'rb') as file:
            series=pickle.load(file)
        return series

    @property
    def sample_frequency(self):
        """Sample frequency of the scope"""
        return 1/self.delta_t

    @property
    def butter_lowpass_polynomials(self):
        """Polynomials for filter"""
        if self._butter_lowpass_polynomials:
            b,a=self._butter_lowpass_polynomials
        else:
            nyquist = 0.5 * self.sample_frequency
            low = self.filter_cutoff/ nyquist
            b, a = butter(self.filter_order, low, btype='low')
        return b, a
