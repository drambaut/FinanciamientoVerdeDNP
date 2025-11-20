
"""PROD_IT2_151025_LimpioLimpio.ipynb


# CARGA
"""



# CARGAR LIBRERIAS
import numpy as np
import pandas as pd
import unicodedata
import os
import csv
import torch
import gc
from transformers import pipeline

# CARGAR MODELO PRE-ENTRENADO
# Auto-selección de dispositivo: GPU si existe, de lo contrario CPU
_device = 0 if (hasattr(torch, "cuda") and torch.cuda.is_available()) else -1
nlp = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",device=_device)

# FUNCION PARA NORMALIZAR TEXTO

def normalizar(texto):
    texto = texto.strip()  # quitar espacios
    texto = unicodedata.normalize("NFKC", texto)  # normaliza acentos compuestos
    return texto

# FUNCION PARA AGREGAR NUEVA COLUMNA EN LA TABLA BIO (META - OBJETIVO - APUESTA)
# De acuerdo a la etiqueta seleccionada como Tema_principal
def reemplazar_ObKM(row, lista1, lista2):
    if row["tema_principal"] in lista1:
        idx = lista1.index(row["tema_principal"])
        return lista2[idx]
    return row["Objetivo KM GBF"]

def reemplazar_ObCDB(row, lista1, lista3):
    if row["tema_principal"] in lista1:
        idx = lista1.index(row["tema_principal"])
        return lista3[idx]
    return row["Objetivo CDB"]
def reemplazar_MetaPAB(row, lista1, lista4):
    if row["tema_principal"] in lista1:
        idx = lista1.index(row["tema_principal"])
        return lista4[idx]
    return row["Meta PAB"]
def reemplazar_ApuestaPAB(row, lista1, lista5):
    if row["tema_principal"] in lista1:
        idx = lista1.index(row["tema_principal"])
        return lista5[idx]
    return row["Apuesta PAB"]

"""## Define las etiquetas para cada una de las categorias y subcategorias..."""

columna_1_BIO = [
    'Integración de la biodiversidad en la planificación espacial',
    'Restauración efectiva del 30% de los ecosistemas degradados',
    'Conservación efectiva de áreas terrestres y marinas',
    'Detener las extinción de especies por causas antropogénicas',
    'Detener la sobreexplotación de especies',
    'Reducir EEI al 50%',
    'Reducir la contaminación al 50%',
    'Reducir los impactos del CC',
    'Manejo sustentable de especies silvestres',
    "Agricultura, pesquerías y forestería sustentable",
    'Restaurar e incrementar SSEE',
    'Incrementar área y calidad de espacios verdes y azules',
    'Reparto justo y equitativo de los beneficios de los RRGG',
    'Integración de la biodiversidad',
    'Monitoréo y transparencia de impactos en la biodiversidad por negocios',
    'Consumo sustentable',
    'Medidas para impactos negativos de los OGM',
    'Eliminar incentivos perversos a la biodiversidad',
    'Incrementar financiamiento',
    'Fortalecer capacidades y acceso a tecnología',
    'Acceso a información y conocimiento',
    'Participación de pueblos indígenas y las comunidades locales',
    'Perspectiva de género'
]
columna_2_BIO = [
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Reducir las amenazas a la biodiversidad',
    'Satisfacer las necesidades de las personas mediante la utilización sostenible y la participación en los beneficios',
    'Satisfacer las necesidades de las personas mediante la utilización sostenible y la participación en los beneficios',
    'Satisfacer las necesidades de las personas mediante la utilización sostenible y la participación en los beneficios',
    'Satisfacer las necesidades de las personas mediante la utilización sostenible y la participación en los beneficios',
    'Satisfacer las necesidades de las personas mediante la utilización sostenible y la participación en los beneficios',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    'Herramientas y soluciones para la implementación y la integración',
    ''
]

columna_3_BIO = [
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'Se mantiene, se aumenta o se restablece la integridad, la conectividad y la resiliencia de todos los ecosistemas, aumentando sustancialmente la superficie de los ecosistemas antes de 2050. Se detiene la extinción de especies amenazadas conocidas causada por la actividad humana y, para 2050, el ritmo y el riesgo de extinción de todas las especies se reduce a la décima parte, y la abundancia de las especies silvestres autóctonas se incrementa a niveles saludables y resilientes; La diversidad genética y el potencial de adaptación de las especies silvestres y domesticadas se mantiene, salvaguardando su potencial de adaptación. ',
    'La biodiversidad se utiliza y gestiona de manera sostenible y las contribuciones de la naturaleza a las personas, entre ellas las funciones y servicios de los ecosistemas, se valoran, se mantienen y se mejoran, restableciéndose aquellas que actualmente están deteriorándose, apoyando el logro del desarrollo sostenible en beneficio de las generaciones actuales y futuras para 2050.',
    'La biodiversidad se utiliza y gestiona de manera sostenible y las contribuciones de la naturaleza a las personas, entre ellas las funciones y servicios de los ecosistemas, se valoran, se mantienen y se mejoran, restableciéndose aquellas que actualmente están deteriorándose, apoyando el logro del desarrollo sostenible en beneficio de las generaciones actuales y futuras para 2050.',
    'La biodiversidad se utiliza y gestiona de manera sostenible y las contribuciones de la naturaleza a las personas, entre ellas las funciones y servicios de los ecosistemas, se valoran, se mantienen y se mejoran, restableciéndose aquellas que actualmente están deteriorándose, apoyando el logro del desarrollo sostenible en beneficio de las generaciones actuales y futuras para 2050.',
    'La biodiversidad se utiliza y gestiona de manera sostenible y las contribuciones de la naturaleza a las personas, entre ellas las funciones y servicios de los ecosistemas, se valoran, se mantienen y se mejoran, restableciéndose aquellas que actualmente están deteriorándose, apoyando el logro del desarrollo sostenible en beneficio de las generaciones actuales y futuras para 2050.',
    'Los beneficios monetarios y no monetarios de la utilización de los recursos genéticos y de la información digital sobre secuencias de recursos genéticos, y de los conocimientos tradicionales asociados a los recursos genéticos, según proceda, se comparten de manera justa y equitativa, y en particular, cuando corresponda, con los pueblos indígenas y las comunidades locales, y se incrementan sustancialmente para 2050, al tiempo que se garantiza que se protegen adecuadamente los conocimientos tradicionales asociados a los recursos genéticos, contribuyendo así a la conservación y utilización sostenible de la diversidad biológica, de conformidad con instrumentos de acceso y participación en los beneficios acordados internacionalmente.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    'Se obtienen medios de implementación adecuados, incluidos recursos financieros, creación de capacidad, cooperación científica y técnica y acceso a la tecnología y su transferencia, para implementar plenamente el Marco Mundial de Biodiversidad de Kunming-Montreal y estos resultan igualmente accesibles para todas las Partes, especialmente las Partes que son países en desarrollo, en particular los países menos adelantados y los pequeños Estados insulares en desarrollo, así como los países con economías en transición, reduciendo progresivamente el déficit de financiación de la biodiversidad de 700.000 millones de dólares de los Estados Unidos al año, y armonizando las corrientes financieras con el Marco Mundial de Biodiversidad de Kunming-Montreal y la Visión de la Diversidad Biológica para 2050.',
    ''
]

columna_4_BIO = [
    'Planificación participativa',
    'Territorios con integridad ecosistémica y modelos regenerativos',
    'Gobernanza de todos los sectores y toda la sociedad',
    'Territorios con integridad ecosistémica y modelos regenerativos',
    'Contaminación, atención de la informalidad y contención de delitos',
    'Potenciar la economía de la biodiversidad',
    'Contaminación, atención de la informalidad y contención de delitos',
    'Planificación participativa',
    'Territorios con integridad ecosistémica y modelos regenerativos',
    'Territorios con integridad ecosistémica y modelos regenerativos',
    'Planificación participativa',
    'Planificación participativa',
    'Potenciar la economía de la biodiversidad',
    'Planificación participativa',
    'Potenciar la economía de la biodiversidad',
    'Potenciar la economía de la biodiversidad',
    'Potenciar la economía de la biodiversidad',
    'Potenciar la economía de la biodiversidad',
    'Modelos Financieros Sostenibles',
    'Potenciar la economía de la biodiversidad',
    'Contaminación, atención de la informalidad y contención de delitos',
    'Gobernanza de todos los sectores y toda la sociedad',
    'Gobernanza de todos los sectores y toda la sociedad',
    ''
]

columna_5_BIO=[
    'Integración y coherencia intersectorial para la gestión territorial de la biodiversidad y la acción climática, como determinantes de la planificación y el ordenamiento',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Gobernanza y corresponsabilidad para la gestión colectiva y biocultural de los territorios para el bienestar de los grupos étnicos y las comunidades locales.',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Atención de la contaminación, la informalidad y contención de los delitos ambientales asociados a los motores de pérdida de la biodiversidad.',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Atención de la contaminación, la informalidad y contención de los delitos ambientales asociados a los motores de pérdida de la biodiversidad.',
    'Integración y coherencia intersectorial para la gestión territorial de la biodiversidad y la acción climática, como determinantes de la planificación y el ordenamiento',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Integración y coherencia intersectorial para la gestión territorial de la biodiversidad y la acción climática, como determinantes de la planificación y el ordenamiento',
    'Integración y coherencia intersectorial para la gestión territorial de la biodiversidad y la acción climática, como determinantes de la planificación y el ordenamiento',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Integración y coherencia intersectorial para la gestión territorial de la biodiversidad y la acción climática, como determinantes de la planificación y el ordenamiento',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Transversal',
    'Impulso a la transición de los modelos productivos hacia la sostenibilidad, la revalorización de la biodiversidad y, la distribución justa y equitativa de los beneficios',
    'Atención de la contaminación, la informalidad y contención de los delitos ambientales asociados a los motores de pérdida de la biodiversidad.',
    'Gobernanza y corresponsabilidad para la gestión colectiva y biocultural de los territorios para el bienestar de los grupos étnicos y las comunidades locales.',
    'Gobernanza y corresponsabilidad para la gestión colectiva y biocultural de los territorios para el bienestar de los grupos étnicos y las comunidades locales.',
    ''
]

categoria_1_CC = ['Minas y energía',"Ambiente y desarrollo sostenible","Agricultura y desarrollo rural",
                  "Transporte", 'Industria, Comercio y Turismo', 'Vivienda, agua y saneamiento',
                  'Salud', 'Control a la Deforestación', 'Transversal', 'Gestión del riesgo y atención de desastres'
                  ]
categoria_1_GRD = ['Reducción del riesgo', 'Manejo de desastres', 'Conocimiento del riesgo',
                   'Gobernanza para la gestión de riesgo de desastres'
                  ]

categoria_2_CC = {
    "Minas y energía": [
        "Generación de energía",
        "Hidrocarburos",
        "Minería",
        "Infraestructura resiliente",
        "Planificación del sistema minero energético",
        "Gestión del entorno",
        "Tecnología, investigación e información"
    ],
    "Ambiente y desarrollo sostenible": [
        "Recurso hídrico",
        "Gestión de los bosques",
        "Biodiversidad",
        "Refrigeración y aire acondicionado (RAC)",
        "Planificación y desarrollo territorial resiliente",
        "Tecnología, investigación e información"
    ],
    "Agricultura y desarrollo rural": [
        "Desarrollo rural",
        "Agricultura",
        "Ganadería bovina",
        "Ganadería no bovina",
        "Tecnología, investigación e información"
    ],
    "Transporte": [
        "Desarrollo urbano y planificación del sector transporte",
        "Transporte Sostenible y Movilidad Eléctrica",
        "Transporte activo y gestión de la demanda",
        "Transporte de carga",
        "Tecnología, investigación e información",
        "Instrumentos económicos y mecanismos"
    ],
    "Industria, Comercio y Turismo": [
        'Eficiencia energética y gestión de la energía (eléctrica y térmica)',
        'Eficiencia en procesos industriales',
        'Logística sostenible',
        'Gestión de aguas residuales industriales',
        'Turismo',
        'Industria resiliente',
        "Tecnología, investigación e información"
    ],
    "Vivienda, agua y saneamiento": [
        'Gestión integral de residuos',
        'Gestión de aguas residuales domésticas',
        'Agua potable',
        'Construcción sostenible',
        "Planificación del sector vivienda, agua y saneamiento básico",
        "Tecnología, investigación e información"
    ],
    "Salud": [
        'Prestación de servicios',
        'Promoción y prevención',
        'Atención de emergencias y desastres'
    ],
    "Control a la Deforestación": [
        'Control a la deforestación',
        "Tecnología, investigación e información",
        'Instrumentos económicos y mecanismos financieros'
    ],
    "Transversal": [
        "Educación, formación y sensibilización",
        'Planificacion',
        "Tecnología, investigación e información",
        'Instrumentos económicos y mecanismos financieros'
    ],
    "Gestión del riesgo y atención de desastres": [
        'Gestión del riesgo asociado a cambio climático'
    ]
}

categoria_2_GRD = {
    "Reducción del riesgo": [
        'Intervención correctiva del riesgo',
        'Intervención prospectiva del riesgo',
        'Protección financiera para el riesgo de desastres'
    ],
    "Manejo de desastres": [
        'Preparación y ejecución de la respuesta',
        'Preparación y ejecución de la recuperación'
    ],
    "Conocimiento del riesgo": [
        'Identificación y caracterización de escenarios, análisis y evaluación del riesgo',
        'Monitoreo del riesgo',
        'Comunicación del riesgo'
    ],
    "Gobernanza para la gestión de riesgo de desastres": [
        "Gobernanza para la gestión de riesgo de desastres"
    ]
}

categoria_BIO = ['Acceso a información y conocimiento',
       'Restauración efectiva del 30% de los ecosistemas degradados',
       'Detener las extinción de especies por causas antropogénicas',
       'Conservación efectiva de áreas terrestres y marinas',
       'Detener la sobreexplotación de especies',
       'Manejo sustentable de especies silvestres',
       'Integración de la biodiversidad en la planificación espacial',
       'Restaurar e incrementar SSEE',
       'Agricultura, pesquerías y forestería sustentable',
       'Incrementar área y calidad de espacios verdes y azules',
       'Integración de la biodiversidad',
       'Fortalecer capacidades y acceso a tecnología',
       'Eliminar incentivos perversos a la biodiversidad']



"""# PROCESO"""

"""
FUNCIONES DE BINARIZACIÓN:
Dependiendo la cantidad de ramificaciones que hayan se binariza de diferente manera

"""

# Binarización a través de percentiles
# Recomendado para cuando hay varias etiquetas! (en este codigo: n>5)
def binarizar_percentil(tabla, keywords, percentil):
    prob_cols = [f"PROB_{kw}" for kw in keywords]

    tabla['umbral'] = tabla[prob_cols].apply(lambda row: np.percentile(row, percentil), axis=1)
    tabla[keywords] = tabla.apply(
        lambda row: [1 if row[f"PROB_{col}"] > row['umbral'] else 0 for col in keywords],
        axis=1, result_type='expand'
    )
    tabla = tabla.drop(columns=['umbral'])
    tabla['etiquetas_sel'] = tabla[keywords].apply(
        lambda row: '-'.join([col for col in keywords if row[col] == 1]),
        axis=1
    )
    return tabla

# Binarización al observar si la diferencia de probabilidades es menor a un umbral
# Tratamiendo cuando hay dos etiquetas!
def binarizar_dos(tabla, keywords, umbral_par):
    col1, col2 = [f"PROB_{kw}" for kw in keywords]
    tabla["diferencia"] = (tabla[col1] - tabla[col2]).abs()
    tabla["cumple"] = np.where(tabla["diferencia"] < umbral_par, "Si", "No")
    tabla = tabla.drop(columns=['diferencia'])

    tabla[keywords] = tabla.apply(
        lambda row: [1 if row["cumple"] == "Si" else 0 for _ in keywords],
        axis=1, result_type='expand'
    )
    tabla = tabla.drop(columns=['cumple'])
    tabla['etiquetas_sel'] = tabla.apply(
        lambda row: '-'.join([col for col in keywords if row[col] == 1])
        if any(row[col] == 1 for col in keywords)
        else row['tema_principal'],
        axis=1
    )
    return tabla

# Binarización al observar si la probabilidad está debajo de un umbral
# Tratamiento como si fuera una sola etiqueta. Cuando no hay muchas etiquetas
def binarizar_uno(tabla, keywords, umbral):
    for kw in keywords:
        tabla[kw] = tabla[f"PROB_{kw}"].apply(lambda x: 1 if x > umbral else 0)
    tabla['etiquetas_sel'] = tabla.apply(
        lambda row: '-'.join([col for col in keywords if row[col] == 1])
        if any(row[col] == 1 for col in keywords)
        else "Ninguno",
        axis=1
    )
    return tabla

# Función para aplicar binarización según sea el caso
# Llama las anteriores funciones dependiendo la cantidad de etiquetas
# Por el momento fijamos el metodo de percentil cuando hay mas de 5 ramificaciones
# y para un numero menor fijamos como binarizar uno.
def aplicar_binarizacion(df_resultados, keywords, umbral_par):
    n = len(keywords)
    if n > 5:
        return binarizar_percentil(df_resultados, keywords, 75)
    #elif n == 4:
    #    return binarizar_percentil(df_resultados, keywords, 50)
    #elif n == 3:
    #    return binarizar_percentil(df_resultados, keywords, 25)
    #elif n == 2:
    #    return binarizar_dos(df_resultados, keywords, umbral_par)
    #elif n == 1:
    #    return binarizar_uno(df_resultados, keywords, umbral_par)
    else:
        return binarizar_uno(df_resultados, keywords, umbral_par)

    return df_resultados

"""
FUNCION DE PROCESAMIENTO DE TABLAS EN CATEGORIAS GRD Y CC
Genera las probabilidades, elige las mayores y binariza
"""

def proceso(col_amarillo, col_bpin, nlp, keywords, umbral_par):
    """
    col_amarillo: Serie o lista de textos (columna AMARILLO).
    col_bpin: Serie/lista de códigos asociados.
    nlp: modelo de clasificación (ej: zero-shot classification de transformers).
    keywords: lista de etiquetas (categorías).
    umbral_par: parámetro para los casos de 1 o 2 keywords.
    """

    objetos = pd.Series(col_amarillo).astype(str).str.replace('.', '', regex=False).tolist()

    resultados = []
    for i, texto in enumerate(objetos):
        #print(i)
        try:
            prob_dict = {}
            # correr keyword por keyword
            for kw in keywords:
                result = nlp(texto, candidate_labels=[kw])
                prob_dict[kw] = result["scores"][0]  # siempre un solo score

            # tema con mayor probabilidad
            tema_principal = max(prob_dict, key=prob_dict.get)
            prob_principal = prob_dict[tema_principal]

            fila = {
                "texto": texto,
                "tema_principal": tema_principal,
                "probabilidad_principal": prob_principal
            }
            for kw in keywords:
                fila[f"PROB_{kw}"] = prob_dict.get(kw, 0)
            resultados.append(fila)

        except Exception as e:
            print(f"Error en fila {i}: {e}")
            continue

        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_resultados = pd.DataFrame(resultados)
    df_resultados['bpin'] = col_bpin

    # aplicar binarización según longitud de keywords
    df_resultados = aplicar_binarizacion(df_resultados, keywords, umbral_par)

    return df_resultados

"""
FUNCION DE PROCESAMIENTO DE TABLAS EN CATEGORIA BIO
Genera las probabilidades, elige las mayores y añade las columnas de META - OBJETIVO - APUESTA
De acuerdo a la etiqueta seleccionada como Tema_principal
* Se hace distinto porque no se ramifica como las demas categorias CC y GRD
"""

def proceso_BIO(col_amarillo, col_bpin, nlp, keywords, umbral_par):
    """
    col_amarillo: Serie o lista de textos (columna AMARILLO).
    col_bpin: Serie/lista de códigos asociados.
    nlp: modelo de clasificación (ej: zero-shot classification de transformers).
    keywords: lista de etiquetas (categorías).
    umbral_par: parámetro para los casos de 1 o 2 keywords.
    """

    objetos = pd.Series(col_amarillo).astype(str).str.replace('.', '', regex=False).tolist()

    resultados = []
    for i, texto in enumerate(objetos):
        #print(i)
        try:
            prob_dict = {}
            # correr keyword por keyword
            for kw in keywords:
                result = nlp(texto, candidate_labels=[kw])
                prob_dict[kw] = result["scores"][0]  # siempre un solo score

            # tema con mayor probabilidad
            tema_principal = max(prob_dict, key=prob_dict.get)
            prob_principal = prob_dict[tema_principal]

            fila = {
                "texto": texto,
                "tema_principal": tema_principal,
                "probabilidad_principal": prob_principal
            }
            resultados.append(fila)

        except Exception as e:
            print(f"Error en fila {i}: {e}")
            continue

        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_resultados = pd.DataFrame(resultados)
    df_resultados['bpin'] = col_bpin

    lista1 = columna_1_BIO
    lista2 = columna_2_BIO
    lista3 = columna_3_BIO
    lista4 = columna_4_BIO
    lista5 = columna_5_BIO

    df_resultados['Objetivo KM GBF']= ['']*len(df_resultados)
    df_resultados['Objetivo CDB']= ['']*len(df_resultados)
    df_resultados['Meta PAB']= ['']*len(df_resultados)
    df_resultados['Apuesta PAB']= ['']*len(df_resultados)

    # Pasar listas como argumentos extra en apply
    df_resultados["Objetivo KM GBF"] = df_resultados.apply(reemplazar_ObKM, axis=1, args=(lista1, lista2))
    df_resultados["Objetivo CDB"] = df_resultados.apply(reemplazar_ObCDB, axis=1, args=(lista1, lista3))
    df_resultados["Meta PAB"] = df_resultados.apply(reemplazar_MetaPAB, axis=1, args=(lista1, lista4))
    df_resultados["Apuesta PAB"] = df_resultados.apply(reemplazar_ApuestaPAB, axis=1, args=(lista1, lista5))

    return df_resultados

"""
FUNCION DE PROCESAMIENTO DE TABLAS EN CATEGORIA BIO
Genera las probabilidades, elige las mayores y binariza de acuerdo a la tabla generada por proceso_BIO
* Se hace distinto porque no se ramifica como las demas categorias CC y GRD
"""

def proceso_BIO2(data,col_amarillo, col_bpin, nlp, keywords, umbral_par):
    """
    col_amarillo: Serie o lista de textos (columna AMARILLO).
    col_bpin: Serie/lista de códigos asociados.
    nlp: modelo de clasificación (ej: zero-shot classification de transformers).
    keywords: lista de etiquetas (categorías).
    umbral_par: parámetro para los casos de 1 o 2 keywords.
    """

    objetos = pd.Series(col_amarillo).astype(str).str.replace('.', '', regex=False).tolist()

    resultados = []
    for i, texto in enumerate(objetos):
        #print(i)
        try:
            prob_dict = {}
            # correr keyword por keyword
            for kw in keywords:
                result = nlp(texto, candidate_labels=[kw])
                prob_dict[kw] = result["scores"][0]  # siempre un solo score

            # tema con mayor probabilidad
            tema_principal = max(prob_dict, key=prob_dict.get)
            prob_principal = prob_dict[tema_principal]

            fila = {
                #"AMARILLO": texto,
                "tema_principal_Eje PNGIBSE": tema_principal,
                "probabilidad_principal_Eje PNGIBSE": prob_principal
            }
            #for kw in keywords:
            #    fila[f"PROB_{kw}"] = prob_dict.get(kw, 0)
            resultados.append(fila)

        except Exception as e:
            print(f"Error en fila {i}: {e}")
            continue

        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.join(data)

    return df_resultados

"""
FUNCIONES DE PROCESAMIENTO DE TABLAS EN CADA RAMIFICACION PARA CATEGORIA CC, GRD Y BIO
"""

# FUNCION PARA PROCESAMIENTO EN SUBCATEGORIAS DE CC Y GRD
def cat1(data_prueba, nlp, nombre, umbral_par, clave):

    if clave == "SI":
      col_bpin = list(data_prueba['bpin'][data_prueba[nombre]==1])
      col_amarillo = list(data_prueba['texto'][data_prueba[nombre]==1])
    else:
      col_bpin = list(data_prueba['bpin'])
      col_amarillo = list(data_prueba['texto'])

    if nombre == "Cambio Climático":
      keywords = categoria_1_CC
    else:
      keywords = categoria_1_GRD
    #print(f"longitud: "+str(len(col_bpin)))
    df_resultados = proceso(col_amarillo, col_bpin, nlp, keywords, umbral_par)
    #print(f"✅ {nombre} Categoria 1")
    return df_resultados


# FUNCION PARA PROCESAMIENTO EN SUBCATEGORIAS DE BIO
# Es diferente porque en esta categoria solo tiene una ramificación
def cat1_BIO(data_prueba, nlp, nombre, umbral_par, clave):
    if clave == "SI":
      col_bpin = list(data_prueba['bpin'][data_prueba[nombre]==1])
      col_amarillo = list(data_prueba['texto'][data_prueba[nombre]==1])
    else:
      col_bpin = list(data_prueba['bpin'])
      col_amarillo = list(data_prueba['texto'])
    keywords = categoria_BIO

    #print(f"longitud: "+str(len(col_bpin)))
    df_resultados = proceso_BIO(col_amarillo, col_bpin, nlp, keywords, umbral_par)
    #print(f"✅ {nombre} Categoria 3")
    return df_resultados

# FUNCION PARA PROCESAMIENTO EN SUBSUBCATEGORIAS DE SUBCATEGORIAS DE CC Y GRD
def cat2(data_prueba, nlp, nombre, umbral_par, clave):
    if nombre == "Cambio Climático":
      lista_etiquetas = categoria_1_CC
    else:
      lista_etiquetas = categoria_1_GRD
    resultados = []
    conteo = 0
    # por una rama...
    for lista in lista_etiquetas:
        conteo += 1
        if clave == "SI":
          col_bpin = list(data_prueba['bpin'][data_prueba[lista]==1])
          col_amarillo = list(data_prueba['texto'][data_prueba[lista]==1])
        else:
          col_bpin = list(data_prueba['bpin'])
          col_amarillo = list(data_prueba['texto'])
        if nombre == "Cambio Climático":
          keywords = categoria_2_CC[lista]
        else:
          keywords = categoria_2_GRD[lista]

        #print(f"longitud: "+str(len(col_bpin)))
        if len(col_bpin)>0:
            df_resultados = proceso(col_amarillo, col_bpin, nlp, keywords, umbral_par)

            # separar columnas fijas y dinámicas
            fijas = df_resultados[['bpin', 'texto']]
            dinamicas = df_resultados.drop(columns=['bpin', 'texto']).add_suffix(f"_{lista}")

            # volver a unir
            df_resultados = pd.concat([fijas, dinamicas], axis=1)


            # indexar por (bpin, AMARILLO) para poder alinear después
            df_resultados = df_resultados.set_index(['bpin', 'texto'])
            df_resultados = df_resultados[~df_resultados.index.duplicated(keep="first")]

            resultados.append(df_resultados)

    if not resultados:
        return pd.DataFrame(columns=['bpin', 'texto'])

    # concatenar por columnas (alineando índices)
    df_final = pd.concat(resultados, axis=1).fillna(0).reset_index()
    #print(f"✅ {nombre} Categoria 2")
    return df_final

# FUNCION PARA PROCESAMIENTO EN SUBCATEGORIAS DE TERCER NIVEL
# Aqui se clasifica unicamente sobre las etiquetas de Mitigacion y Adaptacion
def cat3(data_prueba, nlp, nombre, umbral_par):
    col_bpin = list(data_prueba['bpin'])
    col_amarillo = list(data_prueba['texto'])
    keywords = ['Mitigación', 'Adaptación']
    df_resultados = proceso(col_amarillo, col_bpin, nlp, keywords, umbral_par)
    #print(f"✅ {nombre} Categoria 3")
    return df_resultados

"""
FUNCION QUE PROCESA TODAS LAS RAMIFICACIONES (1er, 2do y 3er nivel) PARA LAS CATEGORIAS CC Y GRD
"""


def pipeline_categorias(carpeta, nlp, nombre, file_name,umbral,data_modulo,clave):
    """
    Corre cat1 -> cat2 -> cat3 en secuencia, exportando e importando automáticamente.

    data     : DataFrame con las categorías originales (la 'base')
    carpeta  : carpeta donde están los archivos y donde se guardarán los resultados
    nlp      : modelo de lenguaje
    nombre   : nombre del módulo
    umbral   : umbral de similitud
    """
    resultados = {}

    # Paso 1: leer modulo_results.xlsx (base)
    # archivo_in = os.path.join(carpeta, "modulo_results.xlsx")
    #data_prueba = pd.read_excel('/content/drive/MyDrive/VERDE/UNA SOLA ETIQUETA/ensayo/LIMPIOS_WEB/intento funciones/por_revisar/union_modulo.xlsx')
    data_prueba = data_modulo
    #print("✅ MODULO", len(data_prueba))

    # -------- CATEGORIA 1 --------
    CC_cat1_results = cat1(data_prueba, nlp, nombre, umbral,clave)
    archivo_cat1 = os.path.join(carpeta, f"{file_name}_cat1_results.xlsx")
    CC_cat1_results.to_excel(archivo_cat1, index=False)
    resultados["cat1"] = CC_cat1_results
    columnas_sel = ["texto", "tema_principal", "probabilidad_principal",'bpin']
    CC_cat1_results[columnas_sel].to_excel(os.path.join(carpeta, f"{file_name}_cat1_results_mayor.xlsx"), index=False)
    #print("✅ Categoria 1")

    # -------- CATEGORIA 2 --------
    data_prueba = pd.read_excel(archivo_cat1)  # volver a leer desde archivo
    CC_cat2_results = cat2(data_prueba, nlp, nombre, umbral,clave)
    archivo_cat2 = os.path.join(carpeta, f"{file_name}_cat2_results.xlsx")
    CC_cat2_results.to_excel(archivo_cat2, index=False)
    resultados["cat2"] = CC_cat2_results


    cols = CC_cat2_results.columns.tolist()

    # Guardamos las columnas que queremos
    cols_a_mantener = []

    for i, col in enumerate(cols):
        if "probabilidad_principal" in col:
            # agregar esta columna
            cols_a_mantener.append(col)
            # agregar la anterior (tema_principal), si existe
            if i > 0:
                cols_a_mantener.append(cols[i-1])

    # Añadir manualmente las columnas que siempre se deben conservar
    cols_a_mantener.extend(["texto", "bpin"])

    # Quitar duplicados y mantener el orden original
    cols_a_mantener = [c for c in cols if c in cols_a_mantener]

    # Crear nuevo DataFrame con solo esas columnas
    df_filtrado = CC_cat2_results[cols_a_mantener]
    df_filtrado.to_excel(os.path.join(carpeta, f"{file_name}_cat2_results_mayor.xlsx"), index=False)
    #print("✅ Categoria 2")

    # -------- CATEGORIA 3 --------

    if nombre == "Cambio Climático":
      data_prueba = pd.read_excel(archivo_cat2)  # volver a leer desde archivo
      CC_cat3_results = cat3(data_prueba, nlp, nombre, umbral)
      archivo_cat3 = os.path.join(carpeta, f"{file_name}_cat3_results.xlsx")
      CC_cat3_results.to_excel(archivo_cat3, index=False)
      resultados["cat3"] = CC_cat3_results

      columnas_sel = ["texto", "tema_principal", "probabilidad_principal",'bpin']
      CC_cat3_results[columnas_sel].to_excel(os.path.join(carpeta, f"{file_name}_cat3_results_mayor.xlsx"), index=False)
    #print("✅ Categoria 2")

    #print("✅ Pipeline completo. Archivos exportados en:", carpeta)
    return resultados

"""## Dejar solo lo Amarillo y elimina columnas"""



"""# Probabilidades y binarizado

* *Si los datos son demasiado grandes es mejor procesarlos por bloques*


"""

def Procesa_CATS(data_prueba, carpeta, clave, umbral=0.25):

  if clave == "SI":

    ###############################################################################
    # PROCESA MODULO -B1
    ###############################################################################

    col_bpin = list(data_prueba['bpin'])
    col_amarillo = list(data_prueba['texto'])
    keywords = ['Biodiversidad', 'Cambio Climático', 'Gestión de Riesgos y Desastres']

    # Procesa y exporta MODULO con probabilidades y binarizado
    modulo_results = proceso(col_amarillo, col_bpin, nlp, keywords, umbral)
    modulo_results.to_excel(os.path.join(carpeta, 'modulo_results.xlsx'), index=False)

    # exporta MODULO unicamente con tema y probabilidad principal
    columnas_sel = ["texto", "tema_principal", "probabilidad_principal",'bpin']
    modulo_results[columnas_sel].to_excel(os.path.join(carpeta, 'modulo_results_mayor.xlsx'), index=False)

  else:
    modulo_results = data_prueba

  ###############################################################################
  # PROCESA CC - B2
  ###############################################################################

  nombre = "Cambio Climático" # Con esto hace la ramificacion unicamente para CC
  file_name = 'CC'
  # Esta funcion procesa y a la vez exporta
  CC_resultados = pipeline_categorias(carpeta, nlp, nombre, file_name, umbral,modulo_results,clave)

  ###############################################################################
  # PROCESA GRD - B3
  ###############################################################################

  nombre = "Gestión de Riesgos y Desastres" # Con esto hace la ramificacion unicamente para GRD
  file_name = 'GRD'

  #Esta funcion procesa y a la vez exporta
  GRD_resultados = pipeline_categorias(carpeta, nlp, nombre, file_name, umbral,modulo_results,clave)

  ###############################################################################
  # PROCESA BIO - B4
  ###############################################################################

  nombre = 'Biodiversidad'
  BIO_cat1_results = cat1_BIO(modulo_results, nlp, nombre, umbral,clave)

  col_bpin = list(BIO_cat1_results['bpin'])
  col_amarillo = list(BIO_cat1_results['texto'])
  keywords = ['Biodiversidad, conservación y cuidado de la naturaleza.',
            'Biodiversidad, gobernanza y creación de valor público.',
            'Biodiversidad, desarrollo económico y calidad de vida.',
            'Biodiversidad, gestión del conocimiento tecnología e información.',
            'Biodiversidad, gestión del riesgo y suministro de servicios ecosistémicos.',
            'Biodiversidad, corresponsabilidad y compromisos globales.']

  BIO_resultados = proceso_BIO2(BIO_cat1_results,col_amarillo, col_bpin, nlp, keywords, umbral)
  BIO_resultados.to_excel(os.path.join(carpeta, 'BIO_cat1_results.xlsx'), index=False)
  # exporta BIO unicamente con tema y probabilidad principal
  #columnas_sel = ["texto", "tema_principal", "probabilidad_principal",'bpin']
  #BIO_resultados[columnas_sel].to_excel(os.path.join(carpeta, 'BIO_results_mayor.xlsx'), index=False)

  return modulo_results, CC_resultados, GRD_resultados, BIO_resultados



# CARGAR DATOS DE PRUEBA
# TEXTO y BPIN
# data_prueba = pd.read_excel('/content/drive/MyDrive/VERDE/RESULTADOS_1_151025/PRUEBA_WEB_concatenado.xlsx') # colocar ruta del excel con los datos y donde la tabla ya contenga "bpin" y "texto"
# data_prueba = data_prueba.rename(columns={'Texto': 'texto'})
# FIJAR LA CARPETA PARA EXPORTAR
# carpeta = "/content/drive/MyDrive/VERDE/RESULTADOS_1_151025/RESULTADOS1" # Colocar ruta de la carpeta a donde los datos se exportaran
# clave = "SI" # En caso de que los datos requieran pasar por proceso de "MODULO" de lo contrario colocar "NO"
# Procesa_CATS(data_prueba, carpeta, clave)



