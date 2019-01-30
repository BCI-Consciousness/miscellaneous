# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:37:42 2018

@author: r3zon
"""

import mne
import os


class BDFPreproc():

    channel_names = ['Fp1', 'AF3',  'F7',   'F3',   'FC1',
                     'FC5', 'T7',   'C3',   'CP1',  'CP5', 
                     'P3',  'Pz',   'PO3',  'O1',   'Oz',
                     'O2',  'P7',   'PO4',  'P4',   'P8',
                     'CP6', 'CP2',  'C4',   'T8',   'FC6',
                     'FC2', 'F4',   'F8',   'AF4',  'Fp2',
                     'Fz',  'Cz',
                     'EXG1', 'EXG2', 'EXG3', 'EXG4',
                     'EXG5', 'EXG6', 'EXG7', 'EXG8',
                     'STI 014']
    
    __l_freq = 2    # low-pass filter freq
    __h_freq = 40    # high-pass filter freq
    
    __path_ica_imgs = '.'
    __path_preprocessed = '.'
    
    
    def __init__(self, bdf_path, ref_type='mastoids'):
        """
        bdf_path - path to .bdf file
        ref_channels - ref type 'mastoids' | 'average'
        """
        self.bdf = mne.io.read_raw_edf(bdf_path, stim_channel=-1) #stim in last channel
        self.path = bdf_path
        self.bdf.load_data()

        self.ref_type = ref_type
        
    def parse_name(self):
        """
        zwraca nazwe badanego obiektu
        """
        file_name = os.path.basename(self.path) # with extension
        file_name_raw = file_name.split('.')[0] # without extension
        return file_name_raw
              
    def __set_refercence(self):
        """
        setting reference type.
        ---
        type = 'average' | 'mastoids'
        """
        ref_channels = 'average'
        if self.ref_type == 'mastoids':
            ref_channels = ['EXG7', 'EXG8']
        self.bdf.set_eeg_reference(ref_channels)  
        
    def __set_bad_channels(self):
        """
        zwraca listę elektrod do wyrzucenia
        """
        self.bdf.info['bads'] = self.channel_names[-9:] #  ostatnie kanały są puste, referencyjne lub z bodźcem, nie potrzebujemy ich do ICA
            
    def __remove_eog(self):
        """
        usuwa artefakty po blinkach na podstawie korelacji komponentu 
        ica i mrugniec
        """
        ch_name = 'EXG1' # kanał referencyjny EOG 
        n_components = 32
        method = 'fastica'
        random_state = 23
        
        eog_epochs = mne.preprocessing.create_eog_epochs(
                self.bdf, 
                ch_name=ch_name) #  znajduje epoki na kanale EOG w których widać blinki
        ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=method,
                random_state=random_state) #  stworzenie obiektu do ICA
        ica.fit(self.bdf) #  dopoasowanie ICA
        
        eog_inds, scores = ica.find_bads_eog(
                eog_epochs,
                ch_name=ch_name) #  eog_inds - najbardziej skorelowane kanały po dekompozycji z kanałem EXG1, scores - wyniki wszystkich korelacji
        
        corelates_figure = ica.plot_scores(
                scores, 
                exclude=eog_inds, 
                show=False) #  rysuje wykres z korelacjami
        
        fig_name = self.parse_name() + '_ica_corelates.png' # ścieżka wykresu z korelacjami
        figure_path = os.path.join(
                self.__path_ica_imgs,
                fig_name)
        corelates_figure.savefig(figure_path) #  zapisz wykres z korelacjami
        ica.exclude.extend(eog_inds) #  wybór kanałów do wyrzucenia przy wykonywaniu odwrotnej ICA
        ica.apply(self.bdf) #  zastosowanie usunięcia kanałów
        
    def preprocess(self):
        """
        wykonuje preprocessing na plikach bdf
        0. ustawienie lokacji elektrod
        1. filtrowanie
        2. wybór złych kanałów
        3. ustawienie referencji
        4. usuwanie artefaktow ocznych
        """
        self.bdf.filter(l_freq=self.__l_freq, h_freq=self.__h_freq)
        self.__set_bad_channels()
        self.__set_refercence()
        self.__remove_eog()

    def save_fif(self):
        """
        zapisuje w formacie mne'owym
        """
        fif_name = self.parse_name() + '.fif'
        fif_path = os.path.join(self.__path_preprocessed,
                                fif_name)
        self.bdf.save(fif_path)
    
    
if __name__ == '__main__':
    # Jak używana była klasa
    path = 'some input pathf'
    bdf = BDFPreproc(path, 'average')
    bdf.preprocess()
    bdf.save_fif('some output path')
    
