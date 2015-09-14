import os
import time
import numpy
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from math import sqrt
import commands
np=numpy

class usbtmc:
    """Simple implementation of a USBTMC device driver, in the style of visa.h"""

    def __init__(self, device):
        self.device = device
        try:
            commands.getoutput('sudo /home/zak/Programs/Bash/chmod_scope.sh')
            self.FILE = os.open(device, os.O_RDWR)
        except OSError:
            raise OSError('Plug in scope and run ' +
                    '"sudo chmod a+rw /dev/usbtmc0"')

    def write(self, command):
        os.write(self.FILE, command);
        #time.sleep(.1)

    def read(self, length = 10000):
        try:
            return os.read(self.FILE, length)
        except OSError:
            return None

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
        self.time_offset=None
        self.delta_t=None
        self.y_reference=None
        self.y_origin=None
        self.y_increment=None
        print "Connected to " + self.name

    def write(self, command):
        """Send an arbitrary command directly to the scope"""
        self.meas.write(command)

    def read(self, length=10000):
        """Read an arbitrary amount of data directly from the scope"""
        return self.meas.read(length)

    def get_channel_data(self, channel_number=1,verbose=False):
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
            if verbose:
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

    def get_scope_settings(self):
        """sets attributes to reflect current scope settings"""
        self.write(":wav:xor?")
        self.time_offset=float(self.read(20))
        self.write(":wav:xinc?")
        self.delta_t=float(self.read(20))
        self.write(":wav:yref?")
        self.y_reference=float(self.read(20))
        self.write(":wav:yor?")
        self.y_origin=float(self.read(20))
        self.write(":wav:yinc?")
        self.y_increment=float(self.read(20))

    def data_criteria_met(self,data):
        """Checks to see if data satisfies the selection criteria"""
        criteria_time=Measurement_Series.criteria_time
        criteria_voltage_min=Measurement_Series.criteria_voltage_min
        criteria_voltage_max=Measurement_Series.criteria_voltage_max
        index=self.time_to_index(criteria_time)
        big_enough= ()

    def set_series_scope_settings(self,series):
        """Sets the scope-related attributes of a measurement_series"""
        series.scope=self
        self.write(":WAV:XOR?")
        series.time_offset=float(self.read(20))
        self.write(":WAV:XINC?")
        series.delta_t=float(self.read(20))
        self.write(":WAV:YREF?")
        series.y_reference=float(self.read(20))
        self.write(":WAV:YOR?")
        series.y_origin=float(self.read(20))
        self.write(":WAV:YINC?")
        series.y_increment=float(self.read(20))

    def get_single_trace(self,channel_number=1,verbose=False,
            get_scope_settings=True):
        """Takes a new trace and returns the channel_data"""
        if get_scope_settings:
            self.get_scope_settings()
        self.write(":SINGLE") #Set trigger to single
        self.write(":TRIGger:STATus?") #Wait until scope takes data
        trigger_status=self.read()
        while trigger_status!='STOP\n':
            time.sleep(0.01)
            self.write(":TRIGger:STATus?")
            trigger_status=self.read()
        try:
        #Retrieve scope data
            data=self.get_channel_data(channel_number,verbose=verbose)
        except BufferError:
            #Call this method recursively until we don't get a BufferError
            if verbose:
                print "Trying to take another trace..."
            data=self.get_single_trace(channel_number)
        if data==None:
            #Happens if we get an OSError which get caught when I included it
            #in the above try/except for some reason
            #Call this method recursively until we don't get an OSError
            data=self.get_single_trace(channel_number)
        return data

    def get_multiple_traces(self,n_traces=10,channel_number=1):
        """Takes multiple traces and returns a Measurement_Series instance"""
        #Get settings for converting times/voltages
        self.get_scope_settings()
        #Prepare object to hold data
        series=Measurement_Series()
        self.set_series_scope_settings(series)
        #Start taking data
        one_data=self.get_single_trace(channel_number)
        data_length=len(one_data)
        series.channel_data=np.expand_dims(one_data,axis=0)
        traces_needed=n_traces-series.n_traces
        interrupted=False
        try:
            while traces_needed>0 and not interrupted:
                #print traces_needed
                data_chunk=numpy.zeros([traces_needed,data_length],
                        dtype=np.uint8)
                try:
                    for j in range(traces_needed):
                        one_data=self.get_single_trace(channel_number,
                                get_scope_settings=False)
                        data_chunk[j]=one_data
                except KeyboardInterrupt:
                    interrupted=True
                all_data=np.vstack([ series.channel_data, data_chunk[:j+1] ])
                series.channel_data=all_data
                series.remove_bad_traces()
                traces_needed=n_traces-series.n_traces
        except KeyboardInterrupt:
            pass
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
            self.measure_tau(n_traces=n_traces,
                    channel_number=channel_number,
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
    left_fit_limit=2e-6#0.7e-6 #time to begin fit
    right_fit_limit=20e-6#15e-6 #time to end fit
    filter_order=10
    filter_cutoff=10e6 #in Hz
    criteria_time=2e-6 #time (sec) at which criteria to keep data is applied
    criteria_voltage_min=1. #Minimum voltage to keep trace (Volts)
    criteria_voltage_max=4. #Maximum voltage to keep trace (Volts)

    def __init__(self):
        """Initializes a Measurement_Series instance"""
        self.scope=None
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
        """Sets attributes to reflect current scope settings

        Now these are usually set right after initialization by the
        AgilentScope instance"""
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
        if self._params.size==0 or self._did_fit==False:
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

    def fit_data(self,remove_bad_traces=True):
        """Fits fit_function to each trace and stores the results

        By default this will call self.remove_bad_traces()"""
        self.remove_bad_traces()
        left_index=self.time_to_index(self.left_fit_limit)
        right_index=self.time_to_index(self.right_fit_limit)
        time_data=self.time_data[left_index:right_index]
        filtered_data=self.filtered_data
        params_array=np.zeros([self.n_traces,3])
        c_left_index=self.time_to_index(20e-6)
        c_right_index=self.time_to_index(30e-6)
        j=0;
        for trace in filtered_data:
            #Guess some initial fitting values
            A=trace[left_index] #amplitude estimate
            tau=2e-6 #lifetime estimate
            c=np.mean(trace[c_left_index:c_right_index]) #offset estimate
            initial_params=(A,tau,c)
            #Do nonlinear fit
            one_params=curve_fit(Measurement_Series.fit_function,
                    time_data,
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
        plt.title(r"Ringdown $\tau$=%1.2f$\mu$s %s Trace %d" %
                (tau*1e6,self.file_name,trace_number))
        plt.ylabel("Voltage (V)")
        plt.xlabel(r"time ($\mu$s)")
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

    def plot_tau_histogram(self):
        """Plots a histogram of the tau values"""
        plt.figure()
        plt.hist(self.tau_array*1e6)
        plt.title(r'Measured $\tau$ values')
        plt.xlabel(r'$\tau$ (us)')
        plt.ylabel('bin count')

    def plot_tau_series(self,bin_size=10):
        """Plots tau values in the order in which they were measured

        This averages groups of bin_size measurements"""
        #Drop data points to make sure its a multiple of bin_size
        n_bins=int(round(self.n_traces/bin_size))
        tau_array=self.tau_array[:(n_bins*bin_size)]
        #Reshape data and take mean across rows
        tau_array.shape=(n_bins,bin_size)
        tau_means=np.mean(tau_array,axis=1)
        tau_uncertainties=np.std(tau_array,axis=1,ddof=1)/np.sqrt(bin_size)
        indices=np.arange(n_bins)+1
        #Convert data to microseconds
        tau_means=tau_means*1e6
        tau_uncertainties=tau_uncertainties*1e6
        #Plot the data
        plt.figure()
        plt.errorbar(indices,tau_means,yerr=tau_uncertainties, fmt='o')
        plt.xlim(0,n_bins+1)
        plt.title(r"$\tau$ measurements")
        plt.xlabel("Measurement Index")
        plt.ylabel(r"$\tau$ ($\mu$s)")

    def remove_bad_traces(self):
        """Removes bad traces from the data set

        Throws out traces if the voltage at time
        Measurement_Series.criteria_time is less than
        Measurement_Series.criteria_voltage_min"""
        channel_data=self.channel_data
        filtered_data=self.filtered_data
        criteria_time=Measurement_Series.criteria_time
        criteria_voltage_min=Measurement_Series.criteria_voltage_min
        criteria_voltage_max=Measurement_Series.criteria_voltage_max
        index=self.time_to_index(criteria_time)
        #Find out whether each row passes the criteria
        big_enough= (filtered_data[:,index]>criteria_voltage_min)
        small_enough= (filtered_data[:,index]<criteria_voltage_max)
        row_useful= np.logical_and(big_enough,small_enough)
        #Now select only those rows and update self.channel_data
        self.channel_data=channel_data[row_useful]
