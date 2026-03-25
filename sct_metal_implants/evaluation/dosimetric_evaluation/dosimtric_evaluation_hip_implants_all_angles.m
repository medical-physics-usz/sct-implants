%% Example: Comparison of a CT and (fake) synthetic CT dose calculation
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2017 the matRad development team. 
% 
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSE.md. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% In this example, we will show:
% (i) how to load DICOM patient data into matRad and modify dosimetric
% objectives and constraints;
% (ii) how to set up a photon dose calculation and optimization based on the real CT;
% (iii) how to translate the optimized plan to be applied to a synthetic CT image; and
% (iv) how to calculate the DVH differences between the plans.

disp('--- headless matRad synCT test started ---');

%% Path 
matRad_cfg = matRad_rc; %If this throws an error, run it from the parent directory first to set the paths

%% Patient Data Import from DICOM  
% General Variable declaration
 
indicators = {'mean','std','max', 'min', 'D_2','D_5', 'D_95', 'D_98'};


% Load Patient Excel
patientInfoFilePath = fullfile(path_excel, "patient_info.xlsx");
patientInfoData = readtable(patientInfoFilePath, "Sheet", 1);

% Define Output Directory
% CSV output paths

dvhOutputDir = fullfile(path_output, "dvh_" + resolution + "mm_all_angles");

% Create the directory if it doesn't exist
if ~exist(dvhOutputDir, 'dir')
    mkdir(dvhOutputDir);
end

% Combined DVH tables of all patients
allDVHReal = table();
allDVHFake = table();
allDVHDiff = table();

% Loop through patients, get correct files with split information

for j = 1:height(patientInfoData)
	
	% Read information from patient-excel
	patientInfoRow = patientInfoData(j, :);
	patientID = string(patientInfoRow.StudyID);
	implantSide = string(patientInfoRow.Side);
	split = string(patientInfoRow.Split);
	
	modelNameWithSplit = string(modelName) + "_" + string(split);

	disp(patientID + ' ' + implantSide + ' ' + split);

	% DICOM paths
	patDirRealCT = fullfile(path_real_data, "CT_real_mask_intersection", patientID);
	patDirFakeCT = fullfile(path_fake_data, modelNameWithSplit, "fake_dicom", patientID); 
	
	%% Copy RT Struct Data to synthetic CT folder
	patDirCST = fullfile(path_real_data, "RTstructs", patientID, "*.dcm");
	copyfile(patDirCST, patDirFakeCT)
	
	%% Load Fake DICOM Data
	impFakeCT = matRad_DicomImporter(patDirFakeCT);
	matRad_importDicom(impFakeCT);
	
	%% Define path to save RTDose for gamma analysis
	RTDoseOutDirFake = fullfile(path_output, "RTDose_" + resolution + "mm_all_angles", "sCT", patientID);
	RTDoseOutDirReal = fullfile(path_output, "RTDose_" + resolution + "mm_all_angles", "rCT", patientID);
	
	if ~exist(RTDoseOutDirFake,'dir')
		mkdir(RTDoseOutDirFake);
	end

	if ~exist(RTDoseOutDirReal,'dir')
		mkdir(RTDoseOutDirReal);
	end
	
	%% Modifying Plan Optimization Objectives  
	
	for i = 1:size(cst, 1)
		structureName = cst{i,2};

		if strcmp(structureName, 'PTV') % PTV
			objective1 = DoseObjectives.matRad_MinDVH;
			objective1.penalty = 1;
			objective1.parameters = {36.25,100}; % Min 36.25 Gy to 100% volume
			
			objective2 = DoseObjectives.matRad_MaxDVH;
			objective2.penalty = 1;
			objective2.parameters = {44.8,0}; % Max 44.8 Gy to 0% volume

			cst{i,6} = {struct(objective1), struct(objective2)};

		elseif strcmp(structureName, 'Ring PTV') % 2cm ring around PTV
			objective = DoseObjectives.matRad_MaxDVH;
			objective.penalty = 1;
			objective.parameters = {34.5, 0}; % Max 34.5 Gray in PTV2cm Ring

			cst{i,6} = {struct(objective)};
		
		elseif strcmp(structureName, 'Prostate')
			objective = DoseObjectives.matRad_MinDVH;
			objective.penalty = 1;
			objective.parameters = {40.0, 95}; % Min 40 Gray in CTV (prostate)
			cst{i,6} = {struct(objective)};
		
		elseif strcmp(structureName, 'Rectum')
			objective1 = DoseObjectives.matRad_MaxDVH;
			objective1.penalty = 1;
			objective1.parameters = {38.0, 0}; % Max 38.0 Gy to 0% volume
			
			objective2 = DoseObjectives.matRad_MeanDose;
			objective2.penalty = 1;
			objective2.parameters = {10}; % Mean dose ≤ 10 Gy
			
			cst{i,6} = {struct(objective1), struct(objective2)};

		elseif strcmp(structureName, 'Bladder')
			objective1 = DoseObjectives.matRad_MaxDVH;
			objective1.penalty = 1;
			objective1.parameters = {40.0, 0}; % Max 40.0 Gy to 0% volume

			objective2 = DoseObjectives.matRad_MaxDVH;
			objective2.penalty = 1;
			objective2.parameters = {38.0, 2}; % Max 38.0 Gy to 2% volume

			objective3 = DoseObjectives.matRad_MeanDose;
			objective3.penalty = 1;
			objective3.parameters = {15}; % Mean dose ≤ 15 Gy
			
			cst{i,6} = {struct(objective1), struct(objective2), struct(objective3)};

		elseif strcmp(structureName, 'Bowel')
			objective = DoseObjectives.matRad_MaxDVH;
			objective.penalty = 1;
			objective.parameters = {30.0, 0}; % Max 30.0 Gy in 0% of volume

			cst{i,6} = {struct(objective)};

		elseif strcmp(structureName, 'Sigma')        
			objective = DoseObjectives.matRad_MaxDVH;
			objective.penalty = 1;
			objective.parameters = {30.0, 0}; % Max 30.0 Gy in 0% of volume

			cst{i,6} = {struct(objective)};

		elseif contains(structureName, 'FemurHead')
			objective = DoseObjectives.matRad_MaxDVH;
			objective.penalty = 1;
			objective.parameters = {10.0, 5}; % Max 10.0 Gy in 5% of volume

			cst{i,6} = {struct(objective)};
		
		elseif strcmp(structureName, 'Body')
			objective = DoseObjectives.matRad_MaxDVH;
			objective.penalty = 1;
			objective.parameters = {18.0, 0}; % Max 18.0 Gy in 0% of volume

			cst{i,6} = {struct(objective)};

		else
			% Mark all other contours as ignored
			cst{i,3} = 'IGNORED';
			cst{i,5}.Priority = 3;
			cst{i,6} = [];
		end
	end
	
	% Verify that the new objectives have been added and are visible in the
	% user interface. We save it for further synthetic CT calculations
	% matRadGUI;
	rtStruct=cst;
	
	%% Treatment Plan
	pln.radiationMode   = 'photons';  
	pln.machine         = 'Generic';
	pln.numOfFractions  = 1;
	
	% Define the biological model used for modeling biological dose
	pln.bioModel = 'none';
	
	% It is possible to request multiple error scenarios for robustness
	pln.multScen = 'nomScen';
	
	% Now we have to set some beam parameters. 
	
	% Assign gantry angles based on implant side
	pln.propStf.gantryAngles = [0 33 66 99 132 165 198 231 264 297 330]; % avoid no hips 

	pln.propStf.couchAngles    = zeros(1,numel(pln.propStf.gantryAngles));
	pln.propStf.bixelWidth     = 5;
	
	% Obtain the number of beams and voxels and calculate the iso-center
	pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
	pln.propStf.isoCenter       = matRad_getIsoCenter(cst,ct,0);
	
	%% Dose calculation settings
	% set resolution of dose calculation and optimization
	pln.propDoseCalc.doseGrid.resolution.x = str2double(resolution); % [mm]
	pln.propDoseCalc.doseGrid.resolution.y = str2double(resolution); % [mm]
	pln.propDoseCalc.doseGrid.resolution.z = str2double(resolution); % [mm]
	
	% Enable sequencing and disable direct aperture optimization (DAO) for now.
	pln.propSeq.runSequencing = 1;
	pln.propOpt.runDAO        = 0;
	
	%% Generate Beam Geometry STF
	% The steering file struct comprises the complete beam geometry along with 
	% ray position, pencil beam positions and energies, source to axis distance (SAD) etc.
	stf = matRad_generateStf(ct,cst,pln);
	
	%% Dose Calculation for fake CT
	% Let's generate dosimetric information by pre-computing dose influence 
	% matrices for unit beamlet intensities for real CT of a patient. Having dose influences available 
	% allows subsequent inverse optimization.
	dij = matRad_calcDoseInfluence(ct,cst,stf,pln);
	
	%% Inverse Optimization for IMRT for fake CT
	% The goal of the fluence optimization is to find a set of beamlet/pencil 
	% beam weights which yield the best possible dose distribution according to
	% the clinical objectives and constraints underlying the radiation 
	% treatment. Once the optimization has finished, trigger once the GUI to 
	% visualize the optimized dose cubes.
	resultGUI = matRad_fluenceOptimization(dij,cst,pln);
	
	% Get the weights of the optimized plan to apply them to the synthetic CT image of the patient later.  
	% Call the matRad_planAnalysis function with the prepared arguments to extract DVH  
	% parameters calculated for the real CT image. 
	weights=resultGUI.w;
	resultGUI = matRad_planAnalysis(resultGUI,ct,cst,stf,pln);
	
	%% Save RT Dose for Gamma Analysis
	
	dcmExp = matRad_DicomExporter();   % ct + cst must be in workspace
	dcmExp.dicomDir = RTDoseOutDirFake;
	
	dcmExp.matRad_exportDicomRTDoses();
	
	%Get the plan parameters
	dvh = resultGUI.dvh;
	qi = resultGUI.qi;
	
	% Save DVH parameters calculated on the real CT to the table  
	% for further comparison with plans calculated on the synthetic CT.  
	% Include the patient number and indicate the CT type as "real"  
	% to facilitate delta calculations between the plans later.
	if matRad_cfg.isMatlab %tables not supported by Octave
	    dvhTableFake=struct2table(qi);
	    % Select only DVH parameters from QI table you are interested in comparison
	    dvhTableFake=dvhTableFake(:,horzcat({'name'},indicators));
	    dvhTableFake.patient= repmat(char(patientID),length(qi),1);
	    dvhTableFake.ct_type = repmat('fake',length(qi),1);
	    % Check DVH table for real CT
	    disp(dvhTableFake);
	end

	%% Now clear the data from the fake CT image, except for the plan parameters
	
	% and load the synthetic (fake) CT image of the same patient.  
	% It is important that your image sets are compatible (i.e., same number of CT slices,  
	% same isocenter position, etc.). We will re-use the structure file from real CT with adjusted objectives 
	% for dose calculations.
	% delete(matRadGUI);
	clear resultGUI ct cst idx qi* dvh dij;
		
	%% Recalculate on real CT
	
	impRealCT = matRad_DicomImporter(patDirRealCT);
	matRad_importDicom(impRealCT);
	cst=rtStruct;
	% Review the exported file and previously identified
	% optimization objectives and constraints. 
	% matRadGUI;
	
	%% Perform the dose calculation for synthetic CT by using the weights of plan, calculated on real CT 
	% (i.e., obtain the dij variable) for the synthetic CT image. 
	resultGUI = matRad_calcDoseDirect(ct,stf,pln,cst, weights);
	resultGUI = matRad_planAnalysis(resultGUI,ct,cst,stf,pln);
	% matRadGUI;
	
	%% Save RT Dose for Gamma Analysis
	
	dcmExp = matRad_DicomExporter();     % ct and cst must exist in workspace NOW
	dcmExp.dicomDir = RTDoseOutDirReal;

	% Write ONLY RTDOSE (uses current workspace vars)
	dcmExp.matRad_exportDicomRTDoses();
	
	
	%Get the plan parameters calculated on synthetic CT
	dvh = resultGUI.dvh;
	qi = resultGUI.qi;
	
	% In the similar fashion, save DVH parameters calculated on the synthetic CT to the table  
	if matRad_cfg.isMatlab  %tables not supported by Octave
	    dvhTableReal=struct2table(qi);
	    % Select the same DVH parameters for further comparison
	    dvhTableReal=dvhTableReal(:,horzcat({'name'},indicators));
	    dvhTableReal.patient= repmat(char(patientID),length(qi),1);
	    dvhTableReal.ct_type = repmat('real',length(qi),1);
	end
	
	%% Calculate the difference in between of DVH calculated on real CT (dvh_table_real) and synthetic CT (dvh_table_fake)
	if matRad_cfg.isMatlab
	    dvhTableDiff = dvhTableFake;
	    for i = 1:height(dvhTableFake)
	        for j = indicators
	            if cell2mat(dvhTableFake{i,'name'}) == cell2mat(dvhTableReal{i,'name'})
	                dvhTableDiff{i,j} = (dvhTableFake{i,j} - dvhTableReal{i,j}) / dvhTableFake{i,j}*100;
	            else
	                dvhTableDiff{i,j} = 'error';
	            end
	        end
	    end
	    disp(dvhTableDiff)
	end

	% Append current patient data to combined tables
	allDVHReal = [allDVHReal; dvhTableReal];
	allDVHFake = [allDVHFake; dvhTableFake];
	allDVHDiff = [allDVHDiff; dvhTableDiff];

	% Clean up before next patient
	clear ct cst dij qi resultGUI dvh weights dvhTableReal dvhTableFake dvhTableDiff;
	
	%% Save DVH Results for all patients for current Patient to CSV - do every iteration in case of error

	writetable(allDVHReal, fullfile(dvhOutputDir, 'dvh_table_real.csv'));
	writetable(allDVHFake, fullfile(dvhOutputDir, 'dvh_table_fake.csv'));
	writetable(allDVHDiff, fullfile(dvhOutputDir, 'dvh_table_diff.csv'));
end










